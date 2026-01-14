"""
Helper functions for listing and managing audio devices.
"""
import pyaudiowpatch as pyaudio
from typing import List, Dict


def list_audio_devices(show_loopback: bool = True) -> List[Dict]:
    """
    List all available audio devices.
    
    Args:
        show_loopback: Include WASAPI loopback devices (speakers as input)
        
    Returns:
        List of device information dictionaries
    """
    p = pyaudio.PyAudio()
    devices = []
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        
        # Add microphone devices
        if info['maxInputChannels'] > 0:
            devices.append({
                'index': i,
                'name': info['name'],
                'type': 'microphone' if 'loopback' not in info['name'].lower() else 'speaker',
                'sample_rate': int(info['defaultSampleRate']),
                'channels': info['maxInputChannels']
            })
    
    p.terminate()
    return devices


def get_default_microphone() -> int:
    """Get the default microphone device index."""
    p = pyaudio.PyAudio()
    default_info = p.get_default_input_device_info()
    index = default_info['index']
    p.terminate()
    return index


def get_default_speaker_loopback() -> int:
    """
    Get the default speaker loopback device index (WASAPI on Windows).
    
    Returns:
        Device index for default speaker loopback
        
    Raises:
        RuntimeError: If WASAPI loopback is not available
    """
    p = pyaudio.PyAudio()
    
    try:
        # Try to get WASAPI host API
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        
        # Get default output device
        if "defaultOutputDevice" not in wasapi_info:
            p.terminate()
            raise RuntimeError("WASAPI found but no default output device")
        
        default_output_index = wasapi_info["defaultOutputDevice"]
        
        if default_output_index < 0:
            p.terminate()
            raise RuntimeError("No default output device available")
        
        # Check if the output device supports loopback (input channels > 0)
        default_speakers = p.get_device_info_by_index(default_output_index)
        
        # For WASAPI loopback, the speaker device should have input channels
        if default_speakers.get("maxInputChannels", 0) > 0:
            index = default_output_index
            p.terminate()
            return index
        else:
            # Try to find any loopback device
            for i in range(p.get_device_count()):
                device = p.get_device_info_by_index(i)
                # Look for devices with "loopback" in name or speakers with input channels
                if (device.get("maxInputChannels", 0) > 0 and 
                    ("loopback" in device["name"].lower() or "stereo mix" in device["name"].lower())):
                    p.terminate()
                    return i
            
            p.terminate()
            raise RuntimeError("WASAPI loopback not available (speakers have no input channels)")
            
    except OSError as e:
        p.terminate()
        raise RuntimeError(f"WASAPI not available on this system: {e}")
    except Exception as e:
        p.terminate()
        raise RuntimeError(f"Failed to get speaker loopback device: {e}")


def print_audio_devices():
    """Print all available audio devices in a readable format."""
    devices = list_audio_devices()
    
    print("\n=== Available Audio Devices ===\n")
    print("Microphones:")
    for dev in devices:
        if dev['type'] == 'microphone':
            print(f"  [{dev['index']}] {dev['name']}")
            print(f"      Sample Rate: {dev['sample_rate']} Hz, Channels: {dev['channels']}")
    
    print("\nSpeakers (Loopback):")
    for dev in devices:
        if dev['type'] == 'speaker':
            print(f"  [{dev['index']}] {dev['name']}")
            print(f"      Sample Rate: {dev['sample_rate']} Hz, Channels: {dev['channels']}")
    
    print()


if __name__ == "__main__":
    # Demo: list all devices
    print_audio_devices()
    
    # Show default devices
    try:
        print(f"Default Microphone Index: {get_default_microphone()}")
    except:
        print("No default microphone found")
    
    try:
        print(f"Default Speaker Loopback Index: {get_default_speaker_loopback()}")
    except Exception as e:
        print(f"Speaker loopback not available: {e}")
