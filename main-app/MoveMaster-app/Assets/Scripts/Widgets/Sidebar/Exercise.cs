using UnityEngine;

[CreateAssetMenu(fileName = "New Exercise", menuName = "Exercises/Add new exercise", order = 1)]
public class Exercise : ScriptableObject
{
    [SerializeField] private string exerciseName;
    [SerializeField] private Sprite exerciseSprite;
    [SerializeField] private Animation exerciseAnimation;
    
    public string ExerciseName => exerciseName;
    public Sprite ExerciseSprite => exerciseSprite;
    public Animation ExerciseAnimation => exerciseAnimation;
}