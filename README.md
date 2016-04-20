# Crosstalk Cancellation
Apply crosstalk cancellation to a binaural audio file  


example usage:  


    python crosstalk_cancellation.py input/audiocheck.wav output/audiocheck.wav  

    
    #if your speakers are 0.8 meters apart, you are sitting 1.5 meters from the speakers, and have a head diamter of 0.215 meters  
    python crosstalk_cancellation.py input/audiocheck.wav output/audiocheck.wav -s 0.8 -l 1.5 -e 0.215 


usage: crosstalk_cancellation.py [-h] [-s SPKR_TO_SPKR] [-l LSTNR_TO_SPKR] [-e EAR_TO_EAR] [-v] audio_path output_path 

positional arguments:
  audio_path            Path to input audio file  
  output                Path to output file  

optional arguments:
  -h, --help            show this help message and exit  
  -s SPKR_TO_SPKR, --spkr_to_spkr SPKR_TO_SPKR  
                        Distance between speakers in meters  
  -l LSTNR_TO_SPKR, --lstnr_to_spkr LSTNR_TO_SPKR  
                        Distance listener is from ceter of speakers in meters  
  -e EAR_TO_EAR, --ear_to_ear EAR_TO_EAR  
                        Distance between ears (diameter of head) in meters  
  -v, --verbose         Print debug messages to stdout  

