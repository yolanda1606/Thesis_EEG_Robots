#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

// Networking headers for Ubuntu (POSIX)
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

#include <Eigen/Dense>
#include <franka/robot.h>
#include <franka/model.h>
#include <franka/gripper.h>
#include <franka/exception.h>

namespace fs = std::filesystem;

// --- Date and Timestamp Helpers ---
std::string get_date_str() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d");
    return ss.str();
}

std::string get_file_timestamp_str() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%H-%M-%S");
    return ss.str();
}

// --- Streamlined Logger Struct (Single Event Column) ---
struct RobotLogRecord {
    double experiment_time;
    int event_trigger; 
    std::string controller_type;
    std::string waypoint_name;
    std::array<double, 7> q;
    std::array<double, 7> dq;
    std::array<double, 16> O_T_EE;
    std::array<double, 7> tau_J;
};

// --- UDP Sender Helper Class (Dual Target) ---
class UdpSender {
private:
    int sockfd;
    struct sockaddr_in eeg_addr;
    struct sockaddr_in video_addr;
    bool initialized = false;

public:
    UdpSender(const std::string& eeg_ip, int eeg_port, const std::string& video_ip, int video_port) {
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd >= 0) {
            memset(&eeg_addr, 0, sizeof(eeg_addr));
            eeg_addr.sin_family = AF_INET;
            eeg_addr.sin_port = htons(eeg_port);
            inet_pton(AF_INET, eeg_ip.c_str(), &eeg_addr.sin_addr);

            memset(&video_addr, 0, sizeof(video_addr));
            video_addr.sin_family = AF_INET;
            video_addr.sin_port = htons(video_port);
            inet_pton(AF_INET, video_ip.c_str(), &video_addr.sin_addr);

            initialized = true;
        } else {
            std::cerr << "Failed to create UDP socket." << std::endl;
        }
    }

    ~UdpSender() {
        if (sockfd >= 0) close(sockfd);
    }

    void send(int trigger_value) {
        if (!initialized || trigger_value == 0) return;
        std::string msg = std::to_string(trigger_value);
        
        sendto(sockfd, msg.c_str(), msg.length(), 0, (struct sockaddr*)&eeg_addr, sizeof(eeg_addr));
        sendto(sockfd, msg.c_str(), msg.length(), 0, (struct sockaddr*)&video_addr, sizeof(video_addr));
        
        std::cout << "[UDP] Sent Trigger: [" << trigger_value << "]" << std::endl;
    }
};

// Data structure for your shapes
struct Shape {
    std::string name;
    int base_trigger; 
    std::array<double, 7> pre_pick;
    std::array<double, 7> pick;
    std::array<double, 7> pre_place;
    std::array<double, 7> place;
};

int main(int argc, char** argv) {
    // Shared memory log buffer 
    std::vector<RobotLogRecord> global_log;
    global_log.reserve(150000); 

    std::string participant_id = "UNKNOWN";
    std::string profile_desc = "UNKNOWN";

    try {
        // --- EXPERIMENT CONFIGURATION MENUS ---
        std::cout << "==========================================\n";
        std::cout << " 0. Enter Participant ID (e.g., P01): ";
        std::cin >> participant_id;

        int mode_input = 0;
        std::cout << "==========================================\n";
        std::cout << " 1. Select Shape Sorter Mode:\n";
        std::cout << "   0: Fault-Free Run (Control)\n";
        std::cout << "   1: Faulty Run (EEG Surprise Factors)\n";
        std::cout << "==========================================\n";
        std::cout << "Choice: ";
        std::cin >> mode_input;
        bool is_faulty = (mode_input == 1);

        int speed_input = 0;
        std::cout << "==========================================\n";
        std::cout << " 2. Select Execution Profile Speed:\n";
        std::cout << "   0: Fast Pace (Standard)\n";
        std::cout << "   1: Slow Pace (Delayed Anticipation)\n";
        std::cout << "==========================================\n";
        std::cout << "Choice: ";
        std::cin >> speed_input;
        
        // --- UPDATED GLOBALLY FASTER MULTIPLIERS ---
        double speed_multiplier = (speed_input == 1) ? 1.8 : 0.8;

        profile_desc = (is_faulty ? "SHAPE_FAULTY" : "SHAPE_CONTROL") + std::string("_") + (speed_input == 1 ? "SLOW" : "FAST");
        std::cout << "\n[CONFIG] Target: " << participant_id << " | Profile: " << profile_desc << "\n\n";

        // --- Initialize Dual UDP Connection ---
        std::string eeg_ip = "10.0.0.2"; 
        int eeg_port = 1000;
        std::string video_ip = "127.0.0.1"; 
        int video_port = 5005;              
        
        UdpSender udp(eeg_ip, eeg_port, video_ip, video_port);

        // --- Robot Initialization ---
        std::string robot_ip = "172.16.0.2";
        franka::Robot robot(robot_ip);
        franka::Gripper gripper(robot_ip);

        double total_experiment_time = 0.0;

        // --- REUSABLE LOGGER LAMBDA (For Gripper Actions & Init) ---
        auto log_action = [&](int trigger, const std::string& action_name) {
            franka::RobotState state = robot.readOnce();
            global_log.push_back({
                total_experiment_time, trigger, "Action/Event", action_name,
                state.q, state.dq, state.O_T_EE, state.tau_J
            });
        };

        // --- REUSABLE MOTION GENERATOR: JOINTS ---
        auto move_joints = [&](const std::array<double, 7>& target_q, double duration, int trigger, const std::string& step_name) {
            std::array<double, 7> start_q;
            double time = 0.0;
            bool first_tick = true;
            bool trigger_logged = false;

            robot.control([&](const franka::RobotState& robot_state, franka::Duration period) -> franka::JointPositions {
                time += period.toSec();
                total_experiment_time += period.toSec();
                
                if (first_tick) {
                    start_q = robot_state.q_d; 
                    first_tick = false;
                }
                
                double u = (time >= duration) ? 1.0 : 0.5 * (1.0 - std::cos(M_PI * time / duration));
                std::array<double, 7> current_q;
                for (size_t i = 0; i < 7; i++) current_q[i] = start_q[i] + u * (target_q[i] - start_q[i]);

                // LOG ONLY ONCE
                int current_log_trigger = 0;
                if (!trigger_logged) {
                    current_log_trigger = trigger;
                    trigger_logged = true; 
                }

                global_log.push_back({
                    total_experiment_time, current_log_trigger, "JointPositions", step_name,
                    robot_state.q, robot_state.dq, robot_state.O_T_EE, robot_state.tau_J
                });

                if (time >= duration) return franka::MotionFinished(franka::JointPositions(current_q));
                return franka::JointPositions(current_q);
            });
        };

        // --- REUSABLE MOTION GENERATOR: CARTESIAN ---
        auto move_cartesian = [&](const Eigen::Vector3d& target_pos, const Eigen::Quaterniond& target_ori, double duration, int trigger, const std::string& step_name) {
            Eigen::Vector3d start_pos;
            Eigen::Quaterniond start_ori;
            double time = 0.0;
            bool first_tick = true;
            bool trigger_logged = false;

            robot.control([&](const franka::RobotState& robot_state, franka::Duration period) -> franka::CartesianPose {
                time += period.toSec();
                total_experiment_time += period.toSec();

                if (first_tick) {
                    Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(robot_state.O_T_EE_c.data()));
                    start_pos = initial_transform.translation();
                    start_ori = Eigen::Quaterniond(initial_transform.rotation());
                    first_tick = false;
                }

                double u = (time >= duration) ? 1.0 : 0.5 * (1.0 - std::cos(M_PI * time / duration));
                Eigen::Vector3d current_pos = start_pos + u * (target_pos - start_pos);
                Eigen::Quaterniond current_ori = start_ori.slerp(u, target_ori);

                Eigen::Affine3d target_transform = Eigen::Affine3d::Identity();
                target_transform.translation() = current_pos;
                target_transform.linear() = current_ori.toRotationMatrix();

                std::array<double, 16> pose_array;
                Eigen::Map<Eigen::Matrix4d> pose_map(pose_array.data());
                pose_map = target_transform.matrix();

                int current_log_trigger = 0;
                if (!trigger_logged) {
                    current_log_trigger = trigger;
                    trigger_logged = true; 
                }

                global_log.push_back({
                    total_experiment_time, current_log_trigger, "CartesianPose", step_name,
                    robot_state.q, robot_state.dq, robot_state.O_T_EE, robot_state.tau_J
                });

                if (time >= duration) return franka::MotionFinished(franka::CartesianPose(pose_array));
                return franka::CartesianPose(pose_array);
            });
        };

        // --- YOUR PYTHON DATA ---
        std::array<double, 7> home_joints = {{-0.0001323, -0.7852356, 0.0002684, -2.3559399, 0.0007338, 1.5711873, 0.7851058}};
        Eigen::Quaterniond down_ori(0.0, 1.0, 0.0, 0.0); 

        std::vector<Shape> shapes = {
            {"CIRCLE", 10,
             {-0.0920499, 0.5526700, 0.0070220, -2.1236928, -0.0227715, 2.6450756, 0.6920618},
             {-0.0910745, 0.6309572, 0.0055707, -2.1017543, -0.0230764, 2.7015183, 0.6927476},
             {0.8709697, 0.6459830, 0.2118781, -1.8536459, -0.1975109, 2.4387715, 1.1894312},
             {0.8773815, 0.6830997, 0.2030048, -1.8484887, -0.2062651, 2.4700643, 1.1949793}},
            {"RECTANGLE", 20,
             {0.0816911, 0.5491361, 0.0872336, -2.1110630, -0.0059721, 2.6425851, 0.9248035},
             {0.0877606, 0.6299139, 0.0821143, -2.0875576, -0.0117393, 2.7000316, 0.9291021},
             {0.6189047, 0.8027075, 0.5731690, -1.7584298, -0.5795325, 2.3339392, -0.038385},
             {0.6189047, 0.8027075, 0.5731690, -1.7584298, -0.5795325, 2.3339392, -0.038385}},
            {"TRIANGLE", 30,
             {0.3149549, 0.6555755, 0.1134322, -1.9254474, -0.1017898, 2.5709333, 1.2158713},
             {0.3217765, 0.7291043, 0.1050727, -1.9009423, -0.1104822, 2.6196166, 1.2219025},
             {0.8419246, 0.7926922, 0.4227393, -1.6460984, -0.3287082, 2.2945599, 1.6922788},
             {0.8429632, 0.8251359, 0.4110818, -1.6356984, -0.3389248, 2.3157003, 1.6874269}},
            {"SQUARE", 40,
             {0.5162047, 0.8150493, 0.1255967, -1.6094433, -0.0977617, 2.4092304, 1.4764988},
             {0.5224169, 0.8901614, 0.1188737, -1.5806791, -0.1020845, 2.4553718, 1.4788992},
             {0.9071025, 0.8091792, 0.4249772, -1.6293220, -0.3366814, 2.3345973, 1.4294769},
             {0.9172860, 0.8419340, 0.4135117, -1.6224530, -0.3475078, 2.3603809, 1.4352874}}
        };

        // --- TASK EXECUTION ---
        std::cout << "Starting Strict Position Control Shape Sorter..." << std::endl;
        
        udp.send(1); 
        log_action(1, "START");

        gripper.move(0.08, 0.1);
        move_joints(home_joints, 4.0 * speed_multiplier, 0, "INIT_HOME");

        for (const auto& item : shapes) {
            std::cout << "\n--- Processing Shape: " << item.name << " ---" << std::endl;
            
            // 1. Pick Phase
            std::cout << "Moving to Pre-Pick..." << std::endl;
            udp.send(item.base_trigger + 1); 
            move_joints(item.pre_pick, 2.5 * speed_multiplier, item.base_trigger + 1, item.name + "_PRE_PICK");
            
            std::cout << "Moving to Pick..." << std::endl;
            udp.send(item.base_trigger + 2);
            move_joints(item.pick, 1.5 * speed_multiplier, item.base_trigger + 2, item.name + "_PICK");
            
            std::cout << "Grasping..." << std::endl;
            udp.send(item.base_trigger + 3);
            log_action(item.base_trigger + 3, item.name + "_GRASP");
            gripper.grasp(0.04, 0.1, 60.0, 0.02, 0.02);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            // Define the safe hover joint position you found
            std::array<double, 7> recover_q = {0.695693, 0.164098, 0.00732913, -1.93717, -0.00220801, 2.0995, 1.48411};

            // 2. Lift/Transition Phase
            if (is_faulty && item.name == "RECTANGLE") {
                std::cout << "    [FAULT INJECTED: Veering directly to Wrong Location!]" << std::endl;
                
                udp.send(80); 
                std::array<double, 7> wrong_q = {-0.0684, 0.2537, -0.8500, -2.2450, 0.2818, 2.7106, 0.7612}; 
                move_joints(wrong_q, 2.5 * speed_multiplier, 80, item.name + "_FAULT_DEVIATION"); 

                std::cout << "    [RECOVERING: Moving to safe mid-air joint pose...]" << std::endl;
                
                udp.send(82); 
                move_joints(recover_q, 2.75 * speed_multiplier, 82, item.name + "_RECOVERY"); 

            } else {
                std::cout << "Lifting (Joint Space)..." << std::endl;
                udp.send(item.base_trigger + 4);
                move_joints(recover_q, 2.5 * speed_multiplier, item.base_trigger + 4, item.name + "_LIFT");

                if (is_faulty && item.name == "SQUARE") {
                    std::cout << "    [FAULT INJECTED: Mid-Air Drop!]" << std::endl;
                    
                    udp.send(81); 
                    log_action(81, item.name + "_FAULT_DROP");
                    gripper.move(0.08, 0.1); 
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            }

            // 3. Place Phase
            std::cout << "Moving to Pre-Place..." << std::endl;
            udp.send(item.base_trigger + 5);
            move_joints(item.pre_place, 2.5 * speed_multiplier, item.base_trigger + 5, item.name + "_PRE_PLACE"); 
            
            std::cout << "Moving to Place..." << std::endl;
            udp.send(item.base_trigger + 6);
            move_joints(item.place, 0.75 * speed_multiplier, item.base_trigger + 6, item.name + "_PLACE");
            
            std::cout << "Releasing..." << std::endl;
            udp.send(item.base_trigger + 7);
            log_action(item.base_trigger + 7, item.name + "_RELEASE");
            gripper.move(0.08, 0.1);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            // 4. Clearance Phase
            std::cout << "Clearance..." << std::endl;
            udp.send(item.base_trigger + 8);
            move_joints(item.pre_place, 1.0 * speed_multiplier, item.base_trigger + 8, item.name + "_CLEARANCE");
        }

        std::cout << "\n>>> Task Complete. Returning Home..." << std::endl;
        udp.send(90); 
        move_joints(home_joints, 3.0 * speed_multiplier, 90, "RETURNING_HOME");
        
        std::cout << "Shape Sorter Successfully Concluded." << std::endl;
        udp.send(99); 
        log_action(99, "END");

    } catch (const franka::Exception& e) {
        std::cerr << "Hardware Exception: " << e.what() << std::endl;
    }

    // --- WRITE BUFFERED LOG TO EXACT FOLDER PATH ---
    std::cout << "\n>>> Writing experiment robot metrics to storage..." << std::endl;
    
    // 1. Build the Target Directory Path dynamically
    std::string base_dir = "/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data/";
    std::string specific_dir = participant_id + "_" + get_date_str() + "/Shape Sorter Observation";
    std::string target_dir = base_dir + specific_dir;

    // 2. Create the entire directory tree
    try {
        fs::create_directories(target_dir);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "ERROR Creating Directories: " << e.what() << std::endl;
    }
    
    // 3. Save the file exactly where you requested
    std::string out_filename = target_dir + "/robot_metrics_" + get_file_timestamp_str() + "_" + profile_desc + ".csv";
    std::ofstream csv_file(out_filename);
    
    if (csv_file.is_open()) {
        csv_file << "Experiment_Time,Event_Trigger,Controller_Type,Waypoint_Stage";
        for(int i=0; i<7; ++i) csv_file << ",q_" << i;
        for(int i=0; i<7; ++i) csv_file << ",dq_" << i;
        for(int i=0; i<16; ++i) csv_file << ",O_T_EE_" << i;
        for(int i=0; i<7; ++i) csv_file << ",tau_J_" << i;
        csv_file << "\n";

        for (const auto& row : global_log) {
            csv_file << row.experiment_time << ","
                     << row.event_trigger << ","
                     << row.controller_type << ","
                     << row.waypoint_name;
            
            for(double val : row.q) csv_file << "," << val;
            for(double val : row.dq) csv_file << "," << val;
            for(double val : row.O_T_EE) csv_file << "," << val;
            for(double val : row.tau_J) csv_file << "," << val;
            csv_file << "\n";
        }
        csv_file.close();
        std::cout << "SUCCESS: Data successfully saved to -> " << out_filename << std::endl;
    } else {
        std::cerr << "ERROR: Failed to save the metrics log file at " << out_filename << std::endl;
    }

    return 0;
}