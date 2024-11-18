using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public class FrameCaptureManager : MonoBehaviour
{
    private IFrameSource _source;
    public bool shouldRunLocally = false;
    
    public UnityEvent<Texture2D> onNewFrameTrigger;
    
    private void Start()
    {
        _source = shouldRunLocally ? new WebcamFrameSource() : new ServerFrameSource();
        _source.SetupCapture();
        StartCoroutine(_source.RunCapture(onNewFrameTrigger));
    }

    private void OnApplicationQuit()
    {
        _source.CleanupCapture();
    }
}
