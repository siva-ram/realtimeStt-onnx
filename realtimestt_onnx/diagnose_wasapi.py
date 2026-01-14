"""
Script to diagnose WASAPI loopback availability and list all audio devices.
"""
import pyaudiowpatch as pyaudio


def diagnose_wasapi():
    """Diagnose WASAPI loopback availability."""
    p = pyaudio.PyAudio()
    
    print("=== WASAPI Loopback Diagnostic ===\n")
    
    # Check for WASAPI
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        print(f"✅ WASAPI found: {wasapi_info['name']}")
        print(f"   Device count: {wasapi_info['deviceCount']}")
        print(f"   Default input: {wasapi_info.get('defaultInputDevice', 'N/A')}")
        print(f"   Default output: {wasapi_info.get('defaultOutputDevice', 'N/A')}\n")
    except OSError:
        print("❌ WASAPI not available on this system\n")
        p.terminate()
        return
    
    # List all devices
    print("=== All Audio Devices ===\n")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']}")
        print(f"  Host API: {p.get_host_api_info_by_index(info['hostApi'])['name']}")
        print(f"  Max Input Channels: {info['maxInputChannels']}")
        print(f"  Max Output Channels: {info['maxOutputChannels']}")
        print(f"  Default Sample Rate: {info['defaultSampleRate']}")
        
        # Check if this could be a loopback device
        if info['maxInputChannels'] > 0 and info['maxOutputChannels'] > 0:
            print("  ⭐ Potential loopback device!")
        
        print()
    
    # Check default output device
    print("=== Default Output Device Check ===\n")
    try:
        default_output = wasapi_info.get('defaultOutputDevice', -1)
        if default_output >= 0:
            output_info = p.get_device_info_by_index(default_output)
            print(f"Default output: {output_info['name']}")
            print(f"Input channels: {output_info['maxInputChannels']}")
            
            if output_info['maxInputChannels'] > 0:
                print("✅ WASAPI loopback AVAILABLE!")
                print(f"   Use device index: {default_output}")
            else:
                print("❌ WASAPI loopback NOT available")
                print("   The default speaker device doesn't support loopback (no input channels)")
                print("\n💡 Solutions:")
                print("   1. Enable 'Stereo Mix' in Windows Sound settings")
                print("   2. Use a virtual audio cable (VB-Audio Cable, etc.)")
                print("   3. Some audio drivers don't support loopback - may need driver update")
        else:
            print("❌ No default output device")
    except Exception as e:
        print(f"❌ Error checking default output: {e}")
    
    print()
    
    # Search for loopback devices
    print("=== Searching for Loopback Devices ===\n")
    found_loopback = False
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name_lower = info['name'].lower()
        
        if ('loopback' in name_lower or 'stereo mix' in name_lower or 
            'what u hear' in name_lower or 'wave out mix' in name_lower):
            print(f"✅ Found loopback device: {info['name']} (index {i})")
            found_loopback = True
    
    if not found_loopback:
        print("❌ No loopback devices found by name")
        print("\n💡 To enable Stereo Mix:")
        print("   1. Right-click speaker icon in taskbar → Sounds")
        print("   2. Go to 'Recording' tab")
        print("   3. Right-click → 'Show Disabled Devices'")
        print("   4. Enable 'Stereo Mix' or similar")
    
    p.terminate()


if __name__ == "__main__":
    diagnose_wasapi()
