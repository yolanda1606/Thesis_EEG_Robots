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

struct ShapePoses {
    std::string name;
    int base_trigger; 
    Eigen::Affine3d pick;
    Eigen::Affine3d pre_pick;
};

int main(int argc, char** argv) {
    // Shared memory log buffer 
    std::vector<RobotLogRecord> global_log;
    global_log.reserve(150000); 

    std::string participant_id = "UNKNOWN";
    std::string profile_desc = "HRI_INTERACTION"; // Fixed profile since no modes

    try {
        // --- EXPERIMENT CONFIGURATION ---
        std::cout << "==========================================\n";
        std::cout << " 0. Enter Participant ID (e.g., P01): ";
        std::cin >> participant_id;
        std::cout << "\n[CONFIG] Target: " << participant_id << " | Profile: " << profile_desc << "\n\n";

        // --- Initialize Dual UDP Connection ---
        std::string eeg_ip = "10.0.0.2"; 
        int eeg_port = 1000;
        std::string video_ip = "127.0.0.1"; 
        int video_port = 5005;              
        
        UdpSender udp(eeg_ip, eeg_port, video_ip, video_port);

        std::string robot_ip = "172.16.0.2";
        franka::Robot robot(robot_ip);
        franka::Gripper gripper(robot_ip);
        franka::Model model = robot.loadModel();

        // 1. Get current robot geometry for the math conversion
        franka::RobotState initial_state = robot.readOnce();
        std::array<double, 16> F_T_EE = initial_state.F_T_EE;
        std::array<double, 16> EE_T_K = initial_state.EE_T_K;

        // 2. Convert Joint Recordings to Cartesian
        std::vector<ShapePoses> shape_list;
        auto convert = [&](std::string name, int base_trigger, std::array<double, 7> q_pick) {
            ShapePoses s;
            s.name = name;
            s.base_trigger = base_trigger;
            auto pose_array = model.pose(franka::Frame::kEndEffector, q_pick, F_T_EE, EE_T_K);
            s.pick = Eigen::Affine3d(Eigen::Matrix4d::Map(pose_array.data()));
            
            s.pick.translation().z() -= 0.012; 
            
            s.pre_pick = s.pick;
            s.pre_pick.translation().z() += 0.10; // 10cm hover
            shape_list.push_back(s);
        };

        convert("CIRCLE",    10, {{-0.091, 0.630, 0.005, -2.101, -0.023, 2.701, 0.692}});
        convert("RECTANGLE", 20, {{0.087, 0.629, 0.082, -2.087, -0.011, 2.700, 0.929}});
        convert("TRIANGLE",  30, {{0.321, 0.729, 0.105, -1.900, -0.110, 2.619, 1.221}});
        convert("SQUARE",    40, {{0.522, 0.890, 0.118, -1.580, -0.102, 2.455, 1.478}});

        // Home Pose for the end of the task
        std::array<double, 7> q_home = {{-0.000, -0.785, 0.000, -2.355, 0.000, 1.571, 0.785}};
        auto home_pose_array = model.pose(franka::Frame::kEndEffector, q_home, F_T_EE, EE_T_K);
        Eigen::Affine3d home_pose(Eigen::Matrix4d::Map(home_pose_array.data()));

        // Participant Handoff Pose
        double z_offset = 0.1134;
        Eigen::Vector3d pos_handoff(0.5794, -0.0582, 0.3649 - z_offset);
        Eigen::Quaterniond ori_handoff(0.1181, -0.7837, 0.5241, -0.3118);

        // Impedance Gains
        double stiffness_val = 400.0;
        Eigen::MatrixXd K(6, 6), D(6, 6);
        K.setZero(); K.topLeftCorner(3, 3) << stiffness_val * Eigen::Matrix3d::Identity();
        K.bottomRightCorner(3, 3) << 15.0 * Eigen::Matrix3d::Identity();
        D.setZero(); D.topLeftCorner(3, 3) << 2.0 * sqrt(stiffness_val) * Eigen::Matrix3d::Identity();
        D.bottomRightCorner(3, 3) << 2.0 * sqrt(15.0) * Eigen::Matrix3d::Identity();

        std::cout << "Starting Silent Cartesian HRI Shape Sorter..." << std::endl;
        
        // --- 1. MANUALLY LOG EXPERIMENT START ---
        udp.send(1); 
        global_log.push_back({
            0.0, 1, "Initialization", "START",
            initial_state.q, initial_state.dq, initial_state.O_T_EE, initial_state.tau_J
        });

        double total_experiment_time = 0.0;

        // --- REUSABLE LOGGER LAMBDA (For Gripper Actions) ---
        auto log_action = [&](int trigger, const std::string& action_name) {
            franka::RobotState state = robot.readOnce();
            global_log.push_back({
                total_experiment_time, trigger, "Action/Event", action_name,
                state.q, state.dq, state.O_T_EE, state.tau_J
            });
        };

        for (const auto& shape : shape_list) {
            std::cout << "\n>>> PIECE: " << shape.name << std::endl;
            
            for (int step = 0; step < 3; ++step) {
                Eigen::Vector3d goal_pos;
                Eigen::Quaterniond goal_ori;
                bool sensing_guard = false;
                
                int active_trigger = 0;
                std::string active_step_name = "";

                if (step == 0) { // PRE-PICK
                    std::cout << "    Moving to Pre-Pick..." << std::endl;
                    active_trigger = shape.base_trigger + 1;
                    active_step_name = shape.name + "_PRE_PICK";
                    udp.send(active_trigger);
                    goal_pos = shape.pre_pick.translation();
                    goal_ori = shape.pre_pick.rotation();
                    gripper.move(0.08, 0.1);
                } else if (step == 1) { // PICK
                    std::cout << "    Moving to Pick..." << std::endl;
                    active_trigger = shape.base_trigger + 2;
                    active_step_name = shape.name + "_PICK";
                    udp.send(active_trigger);
                    goal_pos = shape.pick.translation();
                    goal_ori = shape.pick.rotation();
                } else if (step == 2) { // HANDOFF
                    std::cout << "    Moving to Handoff (Awaiting Participant Pull)..." << std::endl;
                    active_trigger = shape.base_trigger + 4;
                    active_step_name = shape.name + "_HANDOFF";
                    udp.send(active_trigger);
                    goal_pos = pos_handoff;
                    goal_ori = ori_handoff;
                    sensing_guard = true;
                }

                bool first_tick = true;
                bool trigger_logged = false;
                bool piece_taken = false;
                Eigen::Vector3d virtual_pos;
                Eigen::Quaterniond virtual_ori;

                robot.control([&](const franka::RobotState& robot_state, franka::Duration period) -> franka::Torques {
                    total_experiment_time += period.toSec();
                    Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
                    Eigen::Vector3d current_pos(transform.translation());
                    Eigen::Quaterniond current_ori(transform.rotation());

                    if (first_tick) { virtual_pos = current_pos; virtual_ori = current_ori; first_tick = false; }

                    // --- LOGGING ---
                    int log_trig = 0;
                    if (!trigger_logged) {
                        log_trig = active_trigger;
                        trigger_logged = true;
                    }
                    global_log.push_back({
                        total_experiment_time, log_trig, "TorqueImpedance", active_step_name,
                        robot_state.q, robot_state.dq, robot_state.O_T_EE, robot_state.tau_J
                    });

                    double dist_to_goal = (current_pos - goal_pos).norm();
                    
                    // --- HRI FORCE SENSING TRIGGER ---
                    if (sensing_guard && dist_to_goal < 0.03) {
                        Eigen::Map<const Eigen::Matrix<double, 6, 1>> F_ext(robot_state.O_F_ext_hat_K.data());
                        if (F_ext.head(3).norm() > 8.0) { 
                            piece_taken = true;
                            udp.send(shape.base_trigger + 5); 
                            
                            // Exactly log the millisecond the pull happens!
                            global_log.push_back({
                                total_experiment_time, shape.base_trigger + 5, "TorqueImpedance", shape.name + "_HUMAN_PULL",
                                robot_state.q, robot_state.dq, robot_state.O_T_EE, robot_state.tau_J
                            });
                            
                            return franka::MotionFinished(franka::Torques(model.coriolis(robot_state))); 
                        }
                    }

                    double filter = 0.001;
                    virtual_pos = (1.0 - filter) * virtual_pos + filter * goal_pos;
                    virtual_ori = virtual_ori.slerp(filter, goal_ori);

                    Eigen::Matrix<double, 6, 1> error;
                    error.head(3) << current_pos - virtual_pos;
                    if (virtual_ori.coeffs().dot(current_ori.coeffs()) < 0.0) { current_ori.coeffs() << -current_ori.coeffs(); }
                    Eigen::Quaterniond error_q(current_ori.inverse() * virtual_ori);
                    error.tail(3) << error_q.x(), error_q.y(), error_q.z();
                    error.tail(3) << -transform.linear() * error.tail(3);

                    std::array<double, 7> coriolis = model.coriolis(robot_state);
                    std::array<double, 42> jac_array = model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
                    Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jac_array.data());
                    Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());

                    Eigen::VectorXd tau_d = jacobian.transpose() * (-K * error - D * (jacobian * dq)) + Eigen::Map<const Eigen::Matrix<double, 7, 1>>(coriolis.data());
                    std::array<double, 7> tau_array;
                    Eigen::VectorXd::Map(&tau_array[0], 7) = tau_d;

                    if (!sensing_guard && dist_to_goal < 0.015) return franka::MotionFinished(franka::Torques(tau_array));
                    return tau_array;
                });

                if (step == 1) {
                    std::cout << "    [GRASPING]" << std::endl;
                    udp.send(shape.base_trigger + 3);
                    log_action(shape.base_trigger + 3, shape.name + "_GRASP");
                    gripper.grasp(0.03, 0.1, 60.0, 0.02, 0.02);
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                } else if (piece_taken) {
                    std::cout << "    [HUMAN TOOK PIECE - RELEASING]" << std::endl;
                    udp.send(shape.base_trigger + 6);
                    log_action(shape.base_trigger + 6, shape.name + "_RELEASE");
                    gripper.move(0.08, 0.1);
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    break; 
                }
            }
        }

        // --- FINAL RETURN HOME ---
        std::cout << "\n>>> Task Finished. Returning Home..." << std::endl;
        udp.send(90); 

        bool home_tick = true;
        bool home_trigger_logged = false;
        Eigen::Vector3d h_virtual_pos;
        Eigen::Quaterniond h_virtual_ori;
        Eigen::Quaterniond h_goal_ori(home_pose.rotation());

        robot.control([&](const franka::RobotState& robot_state, franka::Duration period) -> franka::Torques {
            total_experiment_time += period.toSec();
            Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
            Eigen::Vector3d current_pos(transform.translation());
            Eigen::Quaterniond current_ori(transform.rotation());

            if (home_tick) { h_virtual_pos = current_pos; h_virtual_ori = current_ori; home_tick = false; }

            int log_trig = 0;
            if (!home_trigger_logged) {
                log_trig = 90;
                home_trigger_logged = true;
            }
            global_log.push_back({
                total_experiment_time, log_trig, "TorqueImpedance", "RETURNING_HOME",
                robot_state.q, robot_state.dq, robot_state.O_T_EE, robot_state.tau_J
            });

            h_virtual_pos = (1.0 - 0.001) * h_virtual_pos + 0.001 * home_pose.translation();
            h_virtual_ori = h_virtual_ori.slerp(0.001, h_goal_ori);

            Eigen::Matrix<double, 6, 1> error;
            error.head(3) << current_pos - h_virtual_pos;
            if (h_virtual_ori.coeffs().dot(current_ori.coeffs()) < 0.0) { current_ori.coeffs() << -current_ori.coeffs(); }
            Eigen::Quaterniond error_q(current_ori.inverse() * h_virtual_ori);
            error.tail(3) << error_q.x(), error_q.y(), error_q.z();
            error.tail(3) << -transform.linear() * error.tail(3);

            std::array<double, 7> coriolis = model.coriolis(robot_state);
            std::array<double, 42> jac_array = model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
            Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jac_array.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());

            Eigen::VectorXd tau_d = jacobian.transpose() * (-K * error - D * (jacobian * dq)) + Eigen::Map<const Eigen::Matrix<double, 7, 1>>(coriolis.data());
            std::array<double, 7> tau_array;
            Eigen::VectorXd::Map(&tau_array[0], 7) = tau_d;

            if ((current_pos - home_pose.translation()).norm() < 0.04) return franka::MotionFinished(franka::Torques(tau_array));
            return tau_array;
        });

        std::cout << "Interactive Task Successfully Concluded." << std::endl;
        udp.send(99); 
        log_action(99, "END");

    } catch (const franka::Exception& e) { std::cerr << e.what() << std::endl; }

    // --- WRITE BUFFERED LOG TO EXACT FOLDER PATH ---
    std::cout << "\n>>> Writing experiment robot metrics to storage..." << std::endl;
    
    std::string base_dir = "/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data/";
    std::string specific_dir = participant_id + "_" + get_date_str() + "/Shape Sorter Interaction";
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