#!/bin/bash

# --- UPDATED ABSOLUTE PATHS ---
BASE_DIR="/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots"
ROBOT_BUILD_DIR="$BASE_DIR/robot_experiments/build"
CAMERA_SCRIPT="$BASE_DIR/camera/vision_node.py"
IMAGE_EXP_DIR="$BASE_DIR/eeg_emotion_experiment"

# Hide annoying TensorFlow/DeepFace GPU warnings
export TF_CPP_MIN_LOG_LEVEL=2 

echo "================================================"
echo "      EEG + ROBOT EXPERIMENT LAUNCHER      "
echo "================================================"

echo -n "Enter Participant ID (e.g., P01): "
read PID

echo ""
echo "Select the experiment to run:"
echo "  --- ROBOT TASKS ---"
echo "  1) Pick and Place"
echo "  2) Shape Sorter Observation"
echo "  3) Shape Sorter Interaction"
echo "  4) Sisyphus"
echo "  5) Stack"
echo "  --- HUMAN ONLY TASKS ---"
echo "  6) Image Experiment (Emotional Baseline)"
echo "  7) Shape Sorter Alone"
echo -n "Choice (1-7): "
read EXP_CHOICE

# Default variables
NEEDS_CONFIG_PROMPT=false
IS_ROBOT_TASK=false
IS_PYTHON_GUI_TASK=false

case $EXP_CHOICE in
    1)
        EXP_DIR="Pick and Place"
        EXEC_NAME="pick_place" 
        NEEDS_CONFIG_PROMPT=true
        PREFIX="PnP"
        IS_ROBOT_TASK=true
        ;;
    2)
        EXP_DIR="Shape Sorter Observation"
        EXEC_NAME="shape_sorter_obs"   
        NEEDS_CONFIG_PROMPT=true
        PREFIX="SHAPE"
        IS_ROBOT_TASK=true
        ;;
    3)
        EXP_DIR="Shape Sorter Interaction"
        EXEC_NAME="shape_sorter_hri" 
        PROFILE="HRI_INTERACTION"  
        IS_ROBOT_TASK=true
        ;;
    4)
        EXP_DIR="Sisyphus"
        EXEC_NAME="sisyphus_task" 
        PROFILE="SISYPHUS_INTERACTION" 
        IS_ROBOT_TASK=true
        ;;
    5)
        EXP_DIR="Stack"
        EXEC_NAME="stack"    
        NEEDS_CONFIG_PROMPT=true
        PREFIX="STACK"
        IS_ROBOT_TASK=true
        ;;
    6)
        EXP_DIR="Image Experiment"
        PROFILE="IMAGE_BASELINE"
        IS_PYTHON_GUI_TASK=true
        ;;
    7)
        EXP_DIR="Shape Sorter Alone"
        PROFILE="SHAPE_ALONE"
        # Leaves both task flags false, triggering the pure human-recording loop
        ;;
    *)
        echo " Invalid choice. Exiting."
        exit 1
        ;;
esac

# Ask for Speed and Fault if required by the C++ code
if [ "$NEEDS_CONFIG_PROMPT" = true ]; then
    echo ""
    echo "Mode Selection for $EXP_DIR:"
    echo "  0) Fault-Free Run (Control)"
    echo "  1) Faulty Run (Surprise Drop / Deviations)"
    echo -n "Choice (0 or 1): "
    read FAULT_MODE
    
    echo ""
    echo "Speed Selection:"
    echo "  0) Fast Pace (Standard)"
    echo "  1) Slow Pace (Delayed Anticipation)"
    echo -n "Choice (0 or 1): "
    read SPEED_MODE

    FAULT_STR="CONTROL"
    if [ "$FAULT_MODE" -eq 1 ]; then FAULT_STR="FAULTY"; fi
    
    SPEED_STR="FAST"
    if [ "$SPEED_MODE" -eq 1 ]; then SPEED_STR="SLOW"; fi

    PROFILE="${PREFIX}_${FAULT_STR}_${SPEED_STR}"
fi

# --- THE EMERGENCY STOP TRAP ---
# If you hit Ctrl+C, it kills the robot, the camera, AND the python GUI safely.
trap 'trap - SIGINT; echo -e "\n STOPPING EXPERIMENT! Halting Processes..."; kill -2 $ROBOT_PID 2>/dev/null; pkill -2 -f "emotion_eeg_exp.py" 2>/dev/null; kill -15 $VISION_PID 2>/dev/null; wait $VISION_PID; exit 1' SIGINT

echo ""
echo "Starting Data Collection for: $EXP_DIR ($PROFILE)"

# 1. Start Python Vision Node (in background)
echo "Starting Vision Node on Port 5005..."
python3 "$CAMERA_SCRIPT" --id "$PID" --exp "$EXP_DIR" --profile "$PROFILE" &
VISION_PID=$!

sleep 4 # Give camera time to warm up pipeline

# 2. Start Logic Branches
if [ "$IS_ROBOT_TASK" = true ]; then
    echo "Executing C++ Robot Node..."
    cd "$ROBOT_BUILD_DIR" || { echo " Could not find build directory"; kill -2 $VISION_PID; exit 1; }

    if [ "$NEEDS_CONFIG_PROMPT" = true ]; then
        echo -e "${PID}\n${FAULT_MODE}\n${SPEED_MODE}" | ./$EXEC_NAME &
    else
        echo -e "${PID}" | ./$EXEC_NAME &
    fi
    
    ROBOT_PID=$!
    wait $ROBOT_PID
    
    echo "Robot path complete! Saving video files..."
    kill -15 $VISION_PID 2>/dev/null
    wait $VISION_PID

elif [ "$IS_PYTHON_GUI_TASK" = true ]; then
    echo "Executing Image Presentation GUI..."
    cd "$IMAGE_EXP_DIR" || { echo " Could not find Image Exp directory"; kill -2 $VISION_PID; exit 1; }
    
    # Activate virtual environment
    source venv/bin/activate 

    # Run the UI
    python3 emotion_eeg_exp.py &
    EXP_PID=$!
    
    # Wait for the python UI to close naturally
    wait $EXP_PID
    
    echo " Image experiment complete! Saving video files..."
    kill -15 $VISION_PID 2>/dev/null
    wait $VISION_PID
    
    # Deactivate virtual environment when done
    deactivate

else
    # Non-Robot, Non-GUI Task Logic (Shape Sorter Alone)
    echo "Human-only task started. The camera is recording."
    echo "Press [Ctrl+C] in this terminal when the participant is finished to save the video."
    
    # Keep script alive until user presses Ctrl+C
    while true; do sleep 1; done
fi

echo "Experiment sequence saved successfully!"