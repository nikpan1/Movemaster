using UnityEngine;
using UnityEngine.UI;

[CreateAssetMenu(fileName = "New Exercise", menuName = "Exercises/Add new exercise", order = 1)]
public class Exercise : ScriptableObject
{
    [SerializeField] private int durationTime;
    [SerializeField] private Sprite exerciseImage;
    
    public int DurationTime => durationTime;
    public Sprite ExerciseImage => exerciseImage;
}
