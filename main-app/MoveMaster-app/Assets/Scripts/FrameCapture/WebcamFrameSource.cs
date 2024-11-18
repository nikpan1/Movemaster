using System.Collections;
using UnityEngine;
using UnityEngine.Events;

public class WebcamFrameSource : IFrameSource
{
    private WebCamTexture _webcam;
    private UnityEvent<Texture2D> _onNewFrameTriggerTexture;
 
    public void SetupCapture()
    {
        if (WebCamTexture.devices.Length == 0)
        {
            Debug.LogError("No webcam devices found.");
            return;
        }
        // @TODO: Probably will change accordingly to TASK-30
        string selectedDevice = WebCamTexture.devices[0].name;
        _webcam = new WebCamTexture(selectedDevice);
    }

    public void CleanupCapture()
    {
        if (_webcam == null) return;
        
        _webcam.Stop();
        _webcam = null;
    }

    public IEnumerator RunCapture(UnityEvent<Texture2D> trigger)
    {
        _onNewFrameTriggerTexture = trigger;

        if (_webcam == null)
        {
            Debug.LogError("Webcam not initialized. Call SetupCapture first.");
            yield break;
        }

        _webcam.Play();

        // Start capturing frames
        while (_webcam.isPlaying)
        {
            Texture2D frame = GetTextureFromWebcam();
            _onNewFrameTriggerTexture?.Invoke(frame);
            yield return null;
        }
    }

    private Texture2D GetTextureFromWebcam()
    {
        Texture2D frame = new Texture2D(_webcam.width, _webcam.height);
        frame.SetPixels(_webcam.GetPixels());
        frame.Apply();
        return frame;
    }
}
