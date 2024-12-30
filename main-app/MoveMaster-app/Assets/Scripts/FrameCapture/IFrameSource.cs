using System.Collections; 
using UnityEngine;
using UnityEngine.Events;

public interface IFrameSource
{
    void SetupCapture(); // @TODO: TASK-30 maybe pass settings?
    void CleanupCapture();
    IEnumerator RunCapture(UnityEvent<Texture2D> triggersTexture2D, UnityEvent<ExerciseInference> triggersExerciseInference);
}