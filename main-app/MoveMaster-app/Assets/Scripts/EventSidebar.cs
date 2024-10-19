using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class EventSidebar : MonoBehaviour
{
    [SerializeField] private GameObject previousExercise;
    [SerializeField] private GameObject currentExercise;
    [SerializeField] private GameObject nextExercise;
    [SerializeField] private Exercise[] exercises;
    [SerializeField] private GameObject endText;
    
    private Image _previousSprite;
    private Image _currentSprite;
    private Image _nextSprite;

    private void Awake()
    {
        _previousSprite = previousExercise.GetComponent<Image>();
        _currentSprite = currentExercise.GetComponent<Image>();
        _nextSprite = nextExercise.GetComponent<Image>();
    }
    private void Start()
    {
        _previousSprite.enabled = false;
        if (exercises.Length < 2)
        {
            Debug.LogWarning("Not enough exercise. Minimum quantity: 2");
            return;
        }

        StartCoroutine(ChangeExercise());
    }

    private IEnumerator ChangeExercise()
    {
        _currentSprite.sprite = exercises[0].ExerciseImage;
        _nextSprite.sprite = exercises[1].ExerciseImage;
        yield return new WaitForSeconds(exercises[0].DurationTime);
        _previousSprite.enabled = true;
        _previousSprite.sprite = exercises[0].ExerciseImage;

        for (int i = 1; i < exercises.Length; i++)
        {
            _currentSprite.sprite = exercises[i].ExerciseImage;
            if (i + 1 > exercises.Length-1)
            {
                _nextSprite.enabled = false;
                break;
            }
            _nextSprite.sprite = exercises[i+1].ExerciseImage;
            yield return new WaitForSeconds(exercises[i].DurationTime);
            _previousSprite.sprite = exercises[i].ExerciseImage;
        }
        
        yield return new WaitForSeconds(exercises[exercises.Length-1].DurationTime);
        _previousSprite.sprite = exercises[exercises.Length-1].ExerciseImage;
        _currentSprite.enabled = false;
        endText.SetActive(true);
    }
}
