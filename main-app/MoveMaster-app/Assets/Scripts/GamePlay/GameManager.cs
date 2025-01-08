using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    private Exercise _currentExercise;
    private void OnEnable()
    {
        GameManagerEvents.ChangeExercise += ChangeExercise;
        GameManagerEvents.CheckExercise += CheckExercise;
    }

    private void OnDisable()
    {
        GameManagerEvents.ChangeExercise -= ChangeExercise;
        GameManagerEvents.CheckExercise -= CheckExercise;
    }

    private void ChangeExercise(Exercise exercise)
    {
        _currentExercise = exercise;
        print(_currentExercise.ExerciseName);
    }

    private void CheckExercise(string detectedExercise, float accuracy)
    {
        if (_currentExercise.ExerciseName == detectedExercise)
        {
            Debug.Log("Good exercise! Accuracy: " + accuracy);
        }
        else
        {
            Debug.Log("Bad exercise!");
        }
    }

    public static class GameManagerEvents
    {
        public static Action<Exercise> ChangeExercise;
        public static Action<string, float> CheckExercise;
    }
}
