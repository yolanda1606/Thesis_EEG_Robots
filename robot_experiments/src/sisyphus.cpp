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
#include <csignal> // Required for intercepting Ctrl+C

// Networking headers for Ubuntu (POSIX)
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>

#include <Eigen/Dense>
#include <franka/robot.h>
#include <franka/model.h>
#include <franka/gripper.h>
#include <franka/exception.h>

namespace fs = std::filesystem;

// --- Global Flag for Safe Infinite Loop Shutdown ---
volatile sig_atomic_t keep_running = 1;

void sigint_handler(int signum) {
    std::cout << "\n\n[SHUTDOWN] Ctrl+C detected! Safely concluding experiment and saving data...\n" << std::endl;
    keep_running = 0;
}

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

enum class TaskState { 
    MOVE_WAIT, PRE_PICK, PICK, POST_PICK, PRE_PLACE, PLACE, POST_PLACE 
};

int main(int argc, char** argv) {
    // Register Ctrl+C Handler
    signal(SIGINT, sigint_handler);

    // Shared memory log buffer (Massive pre-allocation for infinite loop. ~5 mins at 1kHz)
    std::vector<RobotLogRecord> global_log;
    global_log.reserve(300000); 

    std::string participant_id = "UNKNOWN";
    std::string profile_desc = "SISYPHUS_INTERACTION"; 
    double total_experiment_time = 0.0;

    try {
        // --- EXPERIMENT CONFIGURATION ---
        std::cout << "==========================================\n";
        std::cout << " 0. Enter Participant ID (e.g., P01): ";
        std::cin >> participant_id;
        std::cout << "\n[CONFIG] Target: " << participant_id << " | Profile: " << profile_desc << "\n";
        std::cout << "[NOTE] This is an infinite loop task. Press [Ctrl+C] to safely end and save data.\n\n";

        // --- INITIALIZE UDP SENDER ---
        std::string eeg_ip = "10.0.0.2"; 
        int eeg_port = 1000;
        std::string video_ip = "127.0.0.1"; 
        int video_port = 5005; // Video node port for Sisyphus            
        
        UdpSender udp(eeg_ip, eeg_port, video_ip, video_port);

        // --- INITIALIZATION ---
        std::string robot_ip = "172.16.0.2";
        franka::Robot robot(robot_ip); 
        franka::Gripper gripper(robot_ip); 
        franka::Model model = robot.loadModel(); 

        // --- REUSABLE LOGGER LAMBDA (For Gripper Actions) ---
        auto log_action = [&](int trigger, const std::string& action_name) {
            franka::RobotState state = robot.readOnce();
            global_log.push_back({
                total_experiment_time, trigger, "Action/Event", action_name,
                state.q, state.dq, state.O_T_EE, state.tau_J
            });
        };

        // --- SAFETY SETTINGS ---
        robot.setCollisionBehavior(
            {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}, {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
            {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}, {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
            {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}, {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
            {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}, {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}});

        // --- IMPEDANCE GAINS (STIFFNESS) ---
        Eigen::MatrixXd stiffness(6, 6), damping(6, 6);
        stiffness.setZero();
        stiffness.topLeftCorner(3, 3) << 600.0 * Eigen::Matrix3d::Identity(); 
        stiffness.bottomRightCorner(3, 3) << 20.0 * Eigen::Matrix3d::Identity();
        damping.setZero();
        damping.topLeftCorner(3, 3) << 2.0 * sqrt(600.0) * Eigen::Matrix3d::Identity();
        damping.bottomRightCorner(3, 3) << 2.0 * sqrt(20.0) * Eigen::Matrix3d::Identity();

        double z_offset = 0.1134; 
        double clearance = 0.1; 

        // --- COORDINATES ---
        Eigen::Vector3d pos_pick(0.5205, 0.4232, 0.1312 - z_offset);
        Eigen::Quaterniond ori_pick(0.0249, 0.9205, 0.3899, -0.0028);

        Eigen::Vector3d pos_place(0.5205, -0.0, 0.2221 - z_offset);
        Eigen::Quaterniond ori_place(0.0083, -0.9250, -0.3798, -0.0003);

        Eigen::Vector3d pos_wait(0.5252, 0.2075, 0.5284 - z_offset);
        Eigen::Quaterniond ori_wait(0.0260, 0.9232, 0.3835, -0.0036);

        Eigen::Vector3d pos_pre_pick = pos_pick;  pos_pre_pick.z() += clearance;
        Eigen::Vector3d pos_pre_place = pos_place; pos_pre_place.z() += clearance;

        TaskState current_state = TaskState::MOVE_WAIT; 
        std::cout << "Starting Sisyphus Impedance Cycle..." << std::endl;

        // --- 1. MANUALLY LOG EXPERIMENT START ---
        udp.send(1); 
        log_action(1, "START");

        gripper.move(0.08, 0.1); 

        // --- SETUP UDP LISTENERS ---
        auto setup_udp = [](int port) {
            int sock = socket(AF_INET, SOCK_DGRAM, 0);
            sockaddr_in addr;
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            addr.sin_addr.s_addr = INADDR_ANY;
            bind(sock, (struct sockaddr*)&addr, sizeof(addr));
            fcntl(sock, F_SETFL, O_NONBLOCK);
            return sock;
        };

        int vision_sock = setup_udp(5005);
        double latest_stress = 0.0;

        // --- MAIN EXPERIMENT LOOP ---
        while (keep_running) {
            Eigen::Vector3d goal_pos;
            Eigen::Quaterniond goal_ori;
            std::string state_name;
            int state_trigger = 0;
            
            switch(current_state) {
                case TaskState::MOVE_WAIT:  
                    goal_pos = pos_wait; goal_ori = ori_wait; state_name = "WAIT"; state_trigger = 11; break;
                case TaskState::PRE_PICK:   
                    goal_pos = pos_pre_pick; goal_ori = ori_pick; state_name = "PRE_PICK"; state_trigger = 12; break;
                case TaskState::PICK:       
                    goal_pos = pos_pick; goal_ori = ori_pick; state_name = "PICK"; state_trigger = 13; break;
                case TaskState::POST_PICK:  
                    goal_pos = pos_pre_pick; goal_ori = ori_pick; state_name = "LIFTING"; state_trigger = 14; break;
                case TaskState::PRE_PLACE:  
                    goal_pos = pos_pre_place; goal_ori = ori_place; state_name = "PRE_PLACE"; state_trigger = 15; break;
                case TaskState::PLACE:      
                    goal_pos = pos_place; goal_ori = ori_place; state_name = "PLACE"; state_trigger = 16; break;
                case TaskState::POST_PLACE: 
                    goal_pos = pos_pre_place; goal_ori = ori_place; state_name = "CLEARING"; state_trigger = 17; break;
            }

            udp.send(state_trigger); 

            bool is_paused = false;
            double pause_timer = 0.0;
            Eigen::Vector3d virtual_pos;  
            Eigen::Quaterniond virtual_ori;
            bool first_tick = true;
            bool trigger_logged = false;

            // --- THE 1kHz CONTROL LOOP ---
            auto impedance_control = [&](const franka::RobotState& robot_state, franka::Duration period) -> franka::Torques {
                total_experiment_time += period.toSec();
                
                Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
                Eigen::Vector3d current_pos(transform.translation());
                Eigen::Quaterniond current_ori(transform.rotation());
                
                if (first_tick) {
                    virtual_pos = current_pos;
                    virtual_ori = current_ori;
                    first_tick = false;
                }

                int current_log_trigger = 0;
                if (!trigger_logged) {
                    current_log_trigger = state_trigger;
                    trigger_logged = true; 
                }

                // --- COLLISION DETECTION (EEG SAFEGUARD) ---
                Eigen::Map<const Eigen::Matrix<double, 6, 1>> external_forces(robot_state.O_F_ext_hat_K.data());
                double force_magnitude = external_forces.head(3).norm();

                if (force_magnitude > 8.5) { 
                    if (!is_paused) {
                        udp.send(50); // Collision Marker
                        global_log.push_back({
                            total_experiment_time, 50, "TorqueImpedance", "HRI_COLLISION_SPIKE",
                            robot_state.q, robot_state.dq, robot_state.O_T_EE, robot_state.tau_J
                        });
                    }
                    is_paused = true;
                    pause_timer = 0.0;
                } else if (is_paused) {
                    pause_timer += period.toSec();
                    if (pause_timer > 1.5) is_paused = false;
                }

                // Continuous Logging
                global_log.push_back({
                    total_experiment_time, current_log_trigger, "TorqueImpedance", state_name,
                    robot_state.q, robot_state.dq, robot_state.O_T_EE, robot_state.tau_J
                });

                // --- VELOCITY FILTER ---
                double filter = 0.0012; 
                if (is_paused) {
                    virtual_pos = current_pos;
                    virtual_ori = current_ori;
                } else {
                    virtual_pos = (1.0 - filter) * virtual_pos + filter * goal_pos;
                    virtual_ori = virtual_ori.slerp(filter, goal_ori);
                }

                // --- ERROR CALCULATION ---
                Eigen::Matrix<double, 6, 1> error;
                error.head(3) << current_pos - virtual_pos; 

                if (virtual_ori.coeffs().dot(current_ori.coeffs()) < 0.0) { current_ori.coeffs() << -current_ori.coeffs(); }
                Eigen::Quaterniond error_quaternion(current_ori.inverse() * virtual_ori);
                error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
                error.tail(3) << -transform.linear() * error.tail(3);

                // --- TORQUE COMPUTATION ---
                std::array<double, 7> coriolis_array = model.coriolis(robot_state);
                std::array<double, 42> jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
                Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
                Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
                Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());

                Eigen::VectorXd tau_task(7), tau_d(7);
                tau_task << jacobian.transpose() * (-stiffness * error - damping * (jacobian * dq));
                tau_d << tau_task + coriolis;

                std::array<double, 7> tau_d_array;
                Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

                // EXIT CONDITION: Reached target OR user pressed Ctrl+C
                if (!keep_running || (!is_paused && (current_pos - goal_pos).norm() < 0.03)) {
                    return franka::MotionFinished(franka::Torques(tau_d_array));
                }
                return tau_d_array;
            };

            robot.control(impedance_control);

            if (!keep_running) {
                break; // Exit the while loop gracefully to save data
            }

            // --- POST-MOTION LOGIC (Wait/Grasp/Release) ---
            switch(current_state) {
                case TaskState::MOVE_WAIT:  
                    std::cout << "Waiting for ball..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    current_state = TaskState::PRE_PICK; break;
                
                case TaskState::PRE_PICK:   
                    current_state = TaskState::PICK; break;
                
                case TaskState::PICK:       
                    std::cout << "Action: GRASPING" << std::endl;
                    udp.send(20); 
                    log_action(20, "GRASP");
                    gripper.move(0.06, 0.1);
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    current_state = TaskState::POST_PICK; break;
                
                case TaskState::POST_PICK:  
                    current_state = TaskState::PRE_PLACE; break;
                
                case TaskState::PRE_PLACE:  
                    current_state = TaskState::PLACE; break;
                
                case TaskState::PLACE:      
                    std::cout << "Action: RELEASING" << std::endl;
                    udp.send(21); 
                    log_action(21, "RELEASE");
                    gripper.move(0.08, 0.1);
                    current_state = TaskState::POST_PLACE; break;
                
                case TaskState::POST_PLACE: 
                    current_state = TaskState::MOVE_WAIT; break;
            }
        }

        std::cout << "Infinite loop concluded." << std::endl;
        udp.send(99); 
        log_action(99, "END");

    } catch (const franka::Exception& e) {
        std::cerr << "Franka Error: " << e.what() << std::endl;
    }

    // --- WRITE BUFFERED LOG TO EXACT FOLDER PATH ---
    std::cout << "\n>>> Writing experiment robot metrics to storage..." << std::endl;
    
    std::string base_dir = "/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data/";
    std::string specific_dir = participant_id + "_" + get_date_str() + "/Sisyphus";
    std::string target_dir = base_dir + specific_dir;

    try {
        fs::create_directories(target_dir);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "ERROR Creating Directories: " << e.what() << std::endl;
    }
    
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