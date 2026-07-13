#!/usr/bin/env python
# -*- coding: utf-8 -*-

from psychopy import visual, core, event, gui, data
import socket
import os
import random
import datetime  # <-- NEW: Added to match your folder naming conventions

# --- Experiment Settings ---
EXP_NAME = 'Emotion_EEG_Experiment'
EXP_INFO = {'participant': '', 'session': '001'}
DEBUG_MODE = True 

# --- Dual UDP Connection Settings ---
# EEG Destination
UDP_IP = "10.0.0.2"
UDP_PORT = 1000

# Video Node Destination
VIDEO_UDP_IP = "127.0.0.1" 
VIDEO_UDP_PORT = 5005      

# --- Trigger Values ---
TRIGGER_FIXATION = 1
TRIGGER_VALENCE_ONSET = 20  
TRIGGER_AROUSAL_ONSET = 21  

# Practice Triggers
TRIGGER_PRACTICE_IMAGE = 50
TRIGGER_PRACTICE_VALENCE = 60
TRIGGER_PRACTICE_AROUSAL = 61

RATING_DURATION_MAX = 5.0 
# -----------------------------------------------------------

# Initialize UDP socket 
try:
    trigger_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    trigger_address = (UDP_IP, UDP_PORT)
    video_address = (VIDEO_UDP_IP, VIDEO_UDP_PORT) 
    print(f"UDP socket initialized for EEG ({UDP_IP}:{UDP_PORT}) and Video ({VIDEO_UDP_IP}:{VIDEO_UDP_PORT})")
except Exception as e:
    print(f"Error initializing UDP socket: {e}")
    trigger_sock = None

# --- Send trigger to BOTH destinations ---
def send_udp_trigger(value):
    if trigger_sock is None:
        return
    try:
        trigger_message = str(value).encode('utf-8')
        
        # Fire to EEG Laptop
        trigger_sock.sendto(trigger_message, trigger_address)
        # Fire to Video Node
        trigger_sock.sendto(trigger_message, video_address)
        
        if DEBUG_MODE:
            print(f"Sent UDP trigger: {value} to EEG and Video Node")
    except Exception as e:
        print(f"UDP send error for value {value}: {e}")

# --- Dialog box and setup ---
dlg = gui.DlgFromDict(dictionary=EXP_INFO, sortKeys=False, title=EXP_NAME)
if not dlg.OK: core.quit()

# --- AUTOMATED DIRECTORY ROUTING ---
participant_id = EXP_INFO['participant']
now = datetime.datetime.now()
date_folder = now.strftime("%Y-%m-%d")
time_stamp = now.strftime("%H-%M-%S")

# Base folder changed to your new Thesis_EEG_Robots directory
base_dir = '/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data'
target_dir = os.path.join(base_dir, f"{participant_id}_{date_folder}", "Image Experiment")

if not os.path.exists(target_dir): 
    os.makedirs(target_dir)

# Create the file prefix (e.g. emotion_ratings_14-30-00)
filename = f"emotion_ratings_{time_stamp}"
log_file_path = os.path.join(target_dir, filename)

print(f"\n PsychoPy will save ratings to: {target_dir}\n")

# --- Setup PsychoPy Window ---
win = visual.Window(
    size=[1280, 720], fullscr=True, screen=0, winType='pyglet', 
    allowGUI=True, monitor='testMonitor', color='black', units='height'
)
win.mouseVisible = False
globalClock = core.Clock() 

# --- Experiment Components ---
instructions_text = visual.TextStim(win=win, text="Welcome! Press SPACE to start the tutorial.", height=0.05, color='white')
practice_intro_text = visual.TextStim(win=win, 
    text="Let's do a quick practice round.\n\n"
         "Here is how each trial works:\n"
         "1. Focus on the cross (+) in the center of the screen.\n"
         "2. Look at the image that appears.\n"
         "3. Rate how PLEASANT or UNPLEASANT the image made you feel.\n"
         "4. Rate how INTENSE the image made you feel.\n\n"
         "Use the LEFT/RIGHT arrows to move the slider, and SPACE to confirm.\n\n"
         "Press SPACE to begin the practice.", 
    height=0.04, color='white')
practice_end_text = visual.TextStim(win=win, text="Great job! The real experiment will begin now.\n\nPress SPACE when you are ready.", height=0.05, color='white')
end_exp_text = visual.TextStim(win=win, text="Thank you! Press ESC to exit.", height=0.05, color='white')

fixation = visual.TextStim(win=win, text='+', height=0.1, color='white')
image_stim = visual.ImageStim(win=win, name='image_stim', image='sin', pos=(0, 0), size=(0.7, 0.7))

valence_instr = visual.TextStim(win=win, text="VALENCE: How pleasant or unpleasant did the image make you feel?",
                                pos=(0, 0.4), height=0.04, color='white')
arousal_instr = visual.TextStim(win=win, text="AROUSAL: How intense or calm did the image make you feel?",
                                pos=(0, 0.4), height=0.04, color='white')

# VALENCE Rating Slider
valence_rating = visual.Slider(win=win, name='valence_rating',
    size=(0.9, 0.05), pos=(0, -0.3),
    labels=("Very Unpleasant (1)", "", "", "", "", "", "Very Pleasant (7)"), 
    ticks=(1, 2, 3, 4, 5, 6, 7), 
    granularity=1, 
    style=('rating', 'triangleMarker'), 
    lineColor='White', markerColor='Red', labelHeight=0.03, flip=False)

# AROUSAL Rating Slider
arousal_rating = visual.Slider(win=win, name='arousal_rating',
    size=(0.9, 0.05), pos=(0, -0.3),
    labels=("Calm (1)", "", "", "", "", "", "Very Intense (7)"), 
    ticks=(1, 2, 3, 4, 5, 6, 7), 
    granularity=1, 
    style=('rating', 'triangleMarker'), 
    lineColor='White', markerColor='Blue', labelHeight=0.03, flip=False)

# --- Rating Response Function ---
def get_rating_response(rating_stim, instruction_stim, trigger_value):
    rating_stim.reset()
    initial_rating = 4
    rating_stim.markerPos = initial_rating 
    rating_val = initial_rating           
    
    rating_stim.setAutoDraw(True)
    instruction_stim.setAutoDraw(True)
    win.mouseVisible = False 
    
    win.callOnFlip(send_udp_trigger, trigger_value) 
    event.clearEvents()
    
    rating_start_time = globalClock.getTime()
    confirmed = False
    
    while not confirmed and globalClock.getTime() < (rating_start_time + RATING_DURATION_MAX):
        keys = event.getKeys(keyList=['left', 'right', 'space', 'escape'])
        
        for key in keys:
            if key == 'escape':
                win.close(); core.quit()
            elif key == 'left':
                rating_val = max(1, rating_val - 1)
                rating_stim.markerPos = rating_val
            elif key == 'right':
                rating_val = min(7, rating_val + 1)
                rating_stim.markerPos = rating_val
            elif key == 'space': 
                confirmed = True
                rating_rt = globalClock.getTime() - rating_start_time
                break 
                
        if confirmed:
            break
            
        win.flip()
        
    # --- UPDATED FALLBACK LOGIC ---
    if not confirmed:
        # If they didn't press space, use the current rating_val instead of None
        final_rating = rating_val 
        rating_rt = RATING_DURATION_MAX # This already forces RT to 5.0 seconds
    else:
        final_rating = rating_val
        
    rating_stim.setAutoDraw(False)
    instruction_stim.setAutoDraw(False)
    
    return final_rating, rating_rt
        
    rating_stim.setAutoDraw(False)
    instruction_stim.setAutoDraw(False)
    
    return final_rating, rating_rt

# -----------------------------------------------------------
# --- EXPERIMENT LOGIC: CATEGORIES & STRUCTURED TRIGGERS ---

base_image_dir = '/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/eeg_emotion_experiment/stimuli'
categories = ['HAHV', 'HALV', 'LAHV', 'LALV']

trigger_bases = {
    'HAHV': 100,
    'HALV': 200,
    'LAHV': 300,
    'LALV': 400
}

trials = []

# Load images
for category in categories:
    cat_dir = os.path.join(base_image_dir, category)
    
    if not os.path.exists(cat_dir):
        print(f"Warning: Folder {cat_dir} not found! Skipping...")
        continue
        
    image_files = [f for f in os.listdir(cat_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for i, img_file in enumerate(image_files):
        image_path = os.path.join(cat_dir, img_file)
        trigger_val = trigger_bases[category] + (i + 1)
        
        trials.append({
            'stim_id': img_file,        
            'image_path': image_path, 
            'category': category,       
            'trigger': trigger_val      
        })

if not trials:
    print("Error: No images found. Please check your folder paths.")
    core.quit()

# Shuffle all images
random.shuffle(trials)

thisExp = data.ExperimentHandler(name=EXP_NAME, extraInfo=EXP_INFO, savePickle=True, saveWideText=True, dataFileName=log_file_path)

# --- 1. Initial Welcome ---
instructions_text.setAutoDraw(True)
win.flip()
event.waitKeys(keyList=['space'])
instructions_text.setAutoDraw(False)

# --- 2. THE PRACTICE ROUND ---
# Make sure this practice image exists in your folder!
practice_image_path = '/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/eeg_emotion_experiment/stimuli/Sunflower 1.jpg'

practice_intro_text.setAutoDraw(True)
win.flip()
event.waitKeys(keyList=['space'])
practice_intro_text.setAutoDraw(False)

event.clearEvents()
win.callOnFlip(send_udp_trigger, TRIGGER_FIXATION)
fixation.setAutoDraw(True)
win.flip()
core.wait(1.5)
fixation.setAutoDraw(False)

event.clearEvents()
image_stim.image = practice_image_path
win.callOnFlip(send_udp_trigger, TRIGGER_PRACTICE_IMAGE) 
image_stim.setAutoDraw(True)
win.flip()
core.wait(3.0)
image_stim.setAutoDraw(False)

prac_val_rating, prac_val_rt = get_rating_response(valence_rating, valence_instr, TRIGGER_PRACTICE_VALENCE)
prac_arou_rating, prac_arou_rt = get_rating_response(arousal_rating, arousal_instr, TRIGGER_PRACTICE_AROUSAL)

practice_end_text.setAutoDraw(True)
win.flip()
event.waitKeys(keyList=['space'])
practice_end_text.setAutoDraw(False)

# --- 3. MAIN EXPERIMENT LOOP ---
for t_idx, trial in enumerate(trials):
    currentLoop = data.TrialHandler(nReps=1, method='sequential', trialList=[trial], name='trials')
    thisExp.addLoop(currentLoop)

    # Fixation Cross (0.5s)
    event.clearEvents()
    win.callOnFlip(send_udp_trigger, TRIGGER_FIXATION)
    fixation.setAutoDraw(True)
    win.flip()
    core.wait(0.5)
    fixation.setAutoDraw(False)

    # Stimulus Presentation (2.0s)
    event.clearEvents()
    image_stim.image = trial['image_path']
    win.callOnFlip(send_udp_trigger, trial['trigger']) 
    
    image_stim.setAutoDraw(True)
    win.flip()
    core.wait(2.0)
    image_stim.setAutoDraw(False)

    # VALENCE Rating
    val_rating, val_rt = get_rating_response(valence_rating, valence_instr, TRIGGER_VALENCE_ONSET)
    
    # AROUSAL Rating
    arou_rating, arou_rt = get_rating_response(arousal_rating, arousal_instr, TRIGGER_AROUSAL_ONSET)

    # Store trial data
    currentLoop.addData('stim_id', trial['stim_id'])
    currentLoop.addData('category', trial['category']) 
    currentLoop.addData('trigger_sent', trial['trigger']) 
    currentLoop.addData('image_path', trial['image_path'])
    currentLoop.addData('valence_rating', val_rating)
    currentLoop.addData('valence_rt', val_rt)
    currentLoop.addData('arousal_rating', arou_rating)
    currentLoop.addData('arousal_rt', arou_rt)

    thisExp.nextEntry()

# --- End Experiment ---
end_exp_text.setAutoDraw(True)
win.mouseVisible = True
win.flip()
event.waitKeys(keyList=['escape'])

# --- Save data and close ---
thisExp.saveAsWideText(log_file_path + '.csv', delim='auto')
thisExp.saveAsPickle(log_file_path)
if trigger_sock: trigger_sock.close()
win.close()
core.quit()