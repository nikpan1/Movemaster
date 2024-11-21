using UnityEngine;
using UnityEngine.Events;

public class FrameCaptureManager : MonoBehaviour
{
    public bool shouldRunLocally;

    public UnityEvent<Texture2D> onNewFrameTrigger;
    private IFrameSource _source;

    private void Start()
    {
        _source = shouldRunLocally ? new WebcamFrameSource() : new ServerFrameSource();
        _source.SetupCapture();
        StartCoroutine(_source.RunCapture(onNewFrameTrigger));
    }

    private void OnApplicationQuit()
    {
        _source?.CleanupCapture();
    }
}