using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InferencePreview : MonoBehaviour
{
    [SerializeField] private string latestPredictedClass;
    [SerializeField] private float latestPredictedConfidence;

    public void UploadLatestPredictedClass(ExerciseInference inference)
    {
        latestPredictedClass = inference.LatestPredictedClass;
        latestPredictedConfidence = inference.LatestPredictedConfidence;
    }
    
}
