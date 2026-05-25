#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem> // Added for dynamic folder creation

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

struct Waypoint {
    std::string name;
    Eigen::Vector3d pos;
    Eigen::Quaterniond ori;
    double duration; 
    bool grasp_after = false;
    bool release_after = false;
    int trigger_motion = 0; 
    int trigger_action = 0; 
    bool abort_after = false; 
};

struct CubeData {
    std::array<double, 7> pre_pick_q;
    std::array<double, 7> pick_q;
    Eigen::Vector3d pre_place_xyz;
    Eigen::Vector3d place_xyz;
};

int main(int argc, char** argv) {
    // Shared memory log buffer 
    std::vector<RobotLogRecord> global_log;
    global_log.reserve(150000); // Increased slightly for the longer stack task

    std::string participant_id = "UNKNOWN";
    std::string profile_desc = "UNKNOWN";

    try {
        // --- EXPERIMENT CONFIGURATION MENUS ---
        std::cout << "==========================================\n";
        std::cout << " 0. Enter Participant ID (e.g., P01): ";
        std::cin >> participant_id;

        int mode_input = 0;
        std::cout << "==========================================\n";
        std::cout << " 1. Select Stacking Mode:\n";
        std::cout << "   0: Fault-Free Run (Control)\n";
        std::cout << "   1: Faulty Run (Deviations & Mid-Air Drop)\n";
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
        
        // Speed multiplier (Right number -> Fast mode; Left number -> Slow mode)
        double speed_multiplier = (speed_input == 1) ? 1.8 : 0.8;

        profile_desc = (is_faulty ? "STACK_FAULTY" : "STACK_CONTROL") + std::string("_") + (speed_input == 1 ? "SLOW" : "FAST");
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
        franka::Model model = robot.loadModel();

        franka::RobotState initial_state = robot.readOnce();
        std::array<double, 16> F_T_EE = initial_state.F_T_EE;
        std::array<double, 16> EE_T_K = initial_state.EE_T_K;

        double cube_height = 0.04;
        double finger_offset = 0.1034; 
        Eigen::Quaterniond down_ori(0.0, 1.0, 0.0, 0.0);

        std::vector<CubeData> cubes = {
            { // CUBE 1
                {-0.0920499, 0.5526700, 0.0070220, -2.1236928, -0.0227715, 2.6450756, 0.6920618},
                {-0.0910745, 0.6309572, 0.0055707, -2.1017543, -0.0230764, 2.7015183, 0.6927476},
                {0.2412, 0.6043, 0.1569},
                {0.2412, 0.6043, 0.1169}
            },
            { // CUBE 2 (TRAJECTORY DEVIATION fault)
                {0.0816911, 0.5491361, 0.0872336, -2.1110630, -0.0059721, 2.6425851, 0.9248035},
                {0.0877606, 0.6299139, 0.0821143, -2.0875576, -0.0117393, 2.7000316, 0.9291021},
                {0.2412, 0.6043, 0.1569 + cube_height},
                {0.2412, 0.6043, 0.1169 + cube_height}
            },
            { // CUBE 3
                {0.3149549, 0.6555755, 0.1134322, -1.9254474, -0.1017898, 2.5709333, 1.2158713},
                {0.3217765, 0.7291043, 0.1050727, -1.9009423, -0.1104822, 2.6196166, 1.2219025},
                {0.2412, 0.6043, 0.1569 + (2 * cube_height)},
                {0.2412, 0.6043, 0.11 + (2 * cube_height)}
            },
            { // CUBE 4 (MID-AIR DROP fault)
                {0.5162047, 0.8150493, 0.1255967, -1.6094433, -0.0977617, 2.4092304, 1.4764988},
                {0.5224169, 0.8901614, 0.1188737, -1.5806791, -0.1020845, 2.4553718, 1.4788992},
                {0.2412, 0.6043, 0.1569 + (3 * cube_height)},
                {0.2412, 0.6043, 0.10 + (3 * cube_height)} 
            }
        };

        std::array<double, 7> home_pos = {{-0.0001323, -0.7852356, 0.0002684, -2.3559399, 0.0007338, 1.5711873, 0.7851058}};

        // --- BUILD THE MASTER SEQUENCE ---
        std::vector<Waypoint> path;
        for (size_t i = 0; i < cubes.size(); ++i) {
            std::string prefix = "CUBE " + std::to_string(i + 1) + " ";
            int base = (i + 1) * 10; 
            
            auto pre_pick_arr = model.pose(franka::Frame::kEndEffector, cubes[i].pre_pick_q, F_T_EE, EE_T_K);
            Eigen::Affine3d pre_pick_pose(Eigen::Matrix4d::Map(pre_pick_arr.data()));
            
            auto pick_arr = model.pose(franka::Frame::kEndEffector, cubes[i].pick_q, F_T_EE, EE_T_K);
            Eigen::Affine3d pick_pose(Eigen::Matrix4d::Map(pick_arr.data()));

            path.push_back({prefix + "PRE-PICK", pre_pick_pose.translation(), down_ori, 2.5 * speed_multiplier, false, false, base + 1, 0, false});
            path.push_back({prefix + "PICK", pick_pose.translation(), down_ori, 1.5 * speed_multiplier, true, false, base + 2, base + 3, false});
            
            if (is_faulty && i == 1) { // Cube 2 Trajectory Fault
                std::array<double, 7> wrong_q = {-0.0684, 0.2537, -0.8500, -2.2450, 0.2818, 2.7106, 0.7612};
                auto wrong_arr = model.pose(franka::Frame::kEndEffector, wrong_q, F_T_EE, EE_T_K);
                Eigen::Affine3d wrong_pose(Eigen::Matrix4d::Map(wrong_arr.data()));

                path.push_back({prefix + "WRONG LOCATION (FAULT)", wrong_pose.translation(), down_ori, 3.0 * speed_multiplier, false, false, 80, 0, false});
                
                // Note: User previously commented out recovery hover. Leaving out for oblivious fault logic.
            } else {
                path.push_back({prefix + "LIFT", pre_pick_pose.translation(), down_ori, 1.0 * speed_multiplier, false, false, base + 4, 0, false});
            }

            double transit_duration = 2.75 * speed_multiplier;
            int transit_trigger = base + 5;

            if (is_faulty && i == 3) { // Cube 4 Mid-Air Drop Fault
                // Releasing (true), but abort_after is false, so it obliviously keeps going!
                path.push_back({prefix + "INTERMEDIATE (MID-AIR DROP)", {0.4536, 0.3823, 0.25 - finger_offset}, down_ori, transit_duration, false, true, transit_trigger, 81, false});
            } else {
                path.push_back({prefix + "INTERMEDIATE", {0.4536, 0.3823, 0.25 - finger_offset}, down_ori, transit_duration, false, false, transit_trigger, 0, false});
            }
            
            Eigen::Vector3d pre_place = cubes[i].pre_place_xyz;
            pre_place.z() -= finger_offset;
            path.push_back({prefix + "PRE-PLACE", pre_place, down_ori, 2.25 * speed_multiplier, false, false, base + 6, 0, false});

            Eigen::Vector3d place = cubes[i].place_xyz;
            place.z() -= finger_offset;
            
            path.push_back({prefix + "PLACE", place, down_ori, 1.5 * speed_multiplier, false, true, base + 7, base + 8, false});
            path.push_back({prefix + "CLEARANCE", pre_place, down_ori, 1.25 * speed_multiplier, false, false, base + 9, 0, false});
        }

        std::cout << "Starting Strict Position Control Stacking Sequence..." << std::endl;
        
        // --- 1. MANUALLY LOG EXPERIMENT START ---
        udp.send(1); 
        franka::RobotState init_state = robot.readOnce();
        global_log.push_back({
            0.0, 1, "Initialization", "START",
            init_state.q, init_state.dq, init_state.O_T_EE, init_state.tau_J
        });

        gripper.move(0.08, 0.1);

        double total_experiment_time = 0.0;

        // --- EXECUTE MASTER CARTESIAN PATH ---
        for (const auto& point : path) {
            std::cout << ">>> Moving to: " << point.name << std::endl;
            udp.send(point.trigger_motion);

            Eigen::Vector3d start_pos;
            Eigen::Quaterniond start_ori;
            bool first_tick = true;
            bool trigger_logged = false;
            double time = 0.0;
            
            robot.control([&](const franka::RobotState& robot_state, franka::Duration period) -> franka::CartesianPose {
                time += period.toSec();
                total_experiment_time += period.toSec();
                
                if (first_tick) {
                    Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(robot_state.O_T_EE_c.data()));
                    start_pos = initial_transform.translation();
                    start_ori = Eigen::Quaterniond(initial_transform.rotation());
                    first_tick = false;
                }

                double u = (time >= point.duration) ? 1.0 : 0.5 * (1.0 - std::cos(M_PI * time / point.duration));

                Eigen::Vector3d current_target_pos = start_pos + u * (point.pos - start_pos);
                Eigen::Quaterniond current_target_ori = start_ori.slerp(u, point.ori);

                Eigen::Affine3d target_transform = Eigen::Affine3d::Identity();
                target_transform.translation() = current_target_pos;
                target_transform.linear() = current_target_ori.toRotationMatrix();

                std::array<double, 16> pose_array;
                Eigen::Map<Eigen::Matrix4d> pose_map(pose_array.data());
                pose_map = target_transform.matrix();

                // LOG ONLY ONCE LOGIC
                int current_log_trigger = 0;
                if (!trigger_logged) {
                    current_log_trigger = point.trigger_motion;
                    trigger_logged = true; 
                }

                global_log.push_back({
                    total_experiment_time, current_log_trigger, "CartesianPose", point.name,
                    robot_state.q, robot_state.dq, robot_state.O_T_EE, robot_state.tau_J
                });

                if (time >= point.duration) {
                    return franka::MotionFinished(franka::CartesianPose(pose_array));
                }
                return franka::CartesianPose(pose_array);
            });

            // Gripper Action Phase 
            if (point.grasp_after) {
                std::cout << "    [GRASPING]" << std::endl;
                udp.send(point.trigger_action); 

                franka::RobotState state = robot.readOnce();
                global_log.push_back({
                    total_experiment_time, point.trigger_action, "GripperAction", point.name + "_GRASP",
                    state.q, state.dq, state.O_T_EE, state.tau_J
                });

                gripper.grasp(0.04, 0.1, 5.0, 0.02, 0.02);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

            } else if (point.release_after) {
                std::cout << "    [RELEASING / DROPPING]" << std::endl;
                udp.send(point.trigger_action); 

                franka::RobotState state = robot.readOnce();
                global_log.push_back({
                    total_experiment_time, point.trigger_action, "GripperAction", point.name + "_RELEASE",
                    state.q, state.dq, state.O_T_EE, state.tau_J
                });

                gripper.move(0.08, 0.1);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }

            // Kept for structural integrity, though oblivious mode overrides it
            if (point.abort_after) {
                std::cout << "\n>>> FAULT INJECTED: Sequence aborted mid-air." << std::endl;
                break;
            }
        }

        // --- RETURN TO HOME (JOINT CONTROL) ---
        std::cout << "\n>>> Returning to Home..." << std::endl;
        udp.send(90); 

        std::array<double, 7> start_q;
        bool home_tick = true;
        bool home_trigger_logged = false;
        double time_j = 0.0;
        double home_duration = 4.0 * speed_multiplier; 

        robot.control([&](const franka::RobotState& robot_state, franka::Duration period) -> franka::JointPositions {
            time_j += period.toSec();
            total_experiment_time += period.toSec();
            
            if (home_tick) {
                start_q = robot_state.q_d; 
                home_tick = false;
            }

            double u = (time_j >= home_duration) ? 1.0 : 0.5 * (1.0 - std::cos(M_PI * time_j / home_duration));
            
            std::array<double, 7> current_q;
            for(size_t i = 0; i < 7; i++) {
                current_q[i] = start_q[i] + u * (home_pos[i] - start_q[i]);
            }

            int log_trig = 0;
            if (!home_trigger_logged) {
                log_trig = 90; // The home trigger
                home_trigger_logged = true;
            }

            global_log.push_back({
                total_experiment_time, log_trig, "JointPositions", "RETURNING_HOME",
                robot_state.q, robot_state.dq, robot_state.O_T_EE, robot_state.tau_J
            });

            if (time_j >= home_duration) {
                return franka::MotionFinished(franka::JointPositions(current_q));
            }
            return franka::JointPositions(current_q);
        });

        std::cout << "Stacking Task Concluded." << std::endl;
        
        // --- 99. MANUALLY LOG EXPERIMENT END ---
        udp.send(99); 
        franka::RobotState end_state = robot.readOnce();
        global_log.push_back({
            total_experiment_time, 99, "Termination", "END",
            end_state.q, end_state.dq, end_state.O_T_EE, end_state.tau_J
        });

    } catch (const franka::Exception& e) { 
        std::cerr << "Hardware Exception: " << e.what() << std::endl; 
    }

    // --- WRITE BUFFERED LOG TO EXACT FOLDER PATH ---
    std::cout << "\n>>> Writing experiment robot metrics to storage..." << std::endl;
    
    // 1. Build the Target Directory Path dynamically
    std::string base_dir = "/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data/";
    std::string specific_dir = participant_id + "_" + get_date_str() + "/Stack";
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