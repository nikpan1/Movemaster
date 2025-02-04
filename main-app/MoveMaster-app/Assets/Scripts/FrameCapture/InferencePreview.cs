using UnityEngine;

public class InferencePreview : MonoBehaviour
{
    [SerializeField] private string latestPredictedClass;
    [SerializeField] private float latestPredictedConfidence;

    [SerializeField] private EventSidebar eventSidebar;
    private ScoreboardManager scoreboardManager;

    public void UploadLatestPredictedClass(ExerciseInference inference)
    {
        latestPredictedClass = inference.LatestPredictedClass;
        latestPredictedConfidence = inference.LatestPredictedConfidence;
        if (latestPredictedClass == eventSidebar.currentExerciseName)
        {
            ScoreboardManager.Instance.AddValue(1, inference.LatestPredictedConfidence);
            Debug.Log(latestPredictedClass);
        }
    }

    // Method to add 200 to the score (callable from the Inspector)
    [ContextMenu("Add 200 to Score")]
    private void Add200ToScore()
    {
        if (ScoreboardManager.Instance != null)
        {
            ScoreboardManager.Instance.AddValue(200, 10);
            Debug.Log("Added 200 to Score");
        }
        else
        {
            Debug.LogWarning("ScoreboardManager instance is null.");
        }
    }
}