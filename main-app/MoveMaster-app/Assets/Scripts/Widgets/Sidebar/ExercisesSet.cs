using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName = "New Exercises Set", menuName = "Exercises Set/Add new exercises set", order = 1)]
public class ExercisesSet : ScriptableObject
{
    [SerializeField] private List<Exercises> exercisesInSet = new List<Exercises>();
    [SerializeField] private float breakDuration;
    [SerializeField] private DifficultyLevels difficultyLevel;
    
    [System.Serializable]
    public class Exercises
    {
        [SerializeField] private Exercise exercise;
        [SerializeField] int howManySeries;
        [SerializeField] float repetitionCount;
        [SerializeField] bool twoSideExercise;
        
        private float exerciseDuration;
        public Exercise Exercise => exercise;
        public int HowManySeries => howManySeries;
        public float RepetitionCount => repetitionCount;
        public float ExerciseDuration => exerciseDuration;
        public bool TwoSideExercise => twoSideExercise;
    }

    public List<Exercises> ExercisesInSet => exercisesInSet;
    public float BreakDuration => breakDuration;
}
