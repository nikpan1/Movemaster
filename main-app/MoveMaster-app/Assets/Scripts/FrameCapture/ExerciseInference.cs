    public record ExerciseInference
    {
        public string LatestPredictedClass { get; private set; }
        public float LatestPredictedConfidence { get; private set; }

        public ExerciseInference(string latestPredictedClass, float latestPredictedConfidence)
        {
            LatestPredictedClass = latestPredictedClass;
            LatestPredictedConfidence = latestPredictedConfidence;
            GameManager.GameManagerEvents.CheckExercise(latestPredictedClass, latestPredictedConfidence);
        }
    }