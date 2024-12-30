using UnityEngine;
using UnityEngine.Events;

public class FrameCaptureManager : MonoBehaviour
{
    public bool shouldRunLocally;

    public UnityEvent<Texture2D> onNewFrameTrigger;
    public UnityEvent<ExerciseInference> triggersExerciseInference;
    private IFrameSource _source;

    private void Start()
    {
        _source = shouldRunLocally ? new WebcamFrameSource() : new ServerFrameSource();
        _source.SetupCapture();
        StartCoroutine(_source.RunCapture(onNewFrameTrigger, triggersExerciseInference));
    }

    private void OnApplicationQuit()
    {
        _source?.CleanupCapture();
    }
}