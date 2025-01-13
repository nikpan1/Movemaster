using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    private Exercise _currentExercise;
    [SerializeField] private Exercise _previousExercise;
    [SerializeField] private TextMeshProUGUI countdownText;
    private List<float> _currentRepAccuraty = new List<float>();
    private List<float> _repAccuraty = new List<float>();
    private int _countdown = 5;
    private void Start()
    {
        StartCoroutine(CountDownToStart());
    }

    private void OnEnable()
    {
        GameManagerEvents.ChangeExercise += ChangeExercise;
        GameManagerEvents.CheckExercise += CheckExercise;
        GameManagerEvents.NextRepetition += NextRepetition;
        GameManagerEvents.ResetCurrRepAccuraty += ResetCurrRepAccuraty;
    }

    private void OnDisable()
    {
        GameManagerEvents.ChangeExercise -= ChangeExercise;
        GameManagerEvents.CheckExercise -= CheckExercise;
        GameManagerEvents.NextRepetition -= NextRepetition;
        GameManagerEvents.ResetCurrRepAccuraty -= ResetCurrRepAccuraty;
    }

    private void ChangeExercise(Exercise exercise)
    {
        if(_currentExercise != null) _previousExercise = _currentExercise;
        _currentExercise = exercise;
        float points = 0;
        if (_repAccuraty.Count != 0)
        {
            float sum = 0;
            for (int i = 0; i < _repAccuraty.Count; i++)
            {
                sum += _repAccuraty[i];
            }
            points = Mathf.Round((sum/_repAccuraty.Count)*100);
        }
        
        ScoreboardManager.ScoreboardManagerEvents.AddValue((int)points);
        MotivationPanel.MotivationPanelEvents.SetMotivationMessage((int)points, _previousExercise.ExerciseName == "Idle");
        _repAccuraty.Clear();
    }

    private void CheckExercise(string detectedExercise, float accuracy)
    {
        if (_currentExercise == null) return;
        if (_currentExercise.ExerciseName == detectedExercise)
        {
            _currentRepAccuraty.Add(accuracy);
        }
    }

    private void NextRepetition()
    {
        float sum = 0;
        float avarage = 0;
        for (int i = 0; i < _currentRepAccuraty.Count; i++)
        {
            sum += _currentRepAccuraty[i];
        }

        if (_currentRepAccuraty.Count > 0)
        {
            avarage = sum / _currentRepAccuraty.Count;
            _repAccuraty.Add(avarage);
        }
        _currentRepAccuraty.Clear();
    }

    private void ResetCurrRepAccuraty()
    {
        _currentRepAccuraty.Clear();
    }

    private IEnumerator CountDownToStart()
    {
        for (int i = _countdown; i >= 0; i--)
        {
            if (i != 0)
            {
                countdownText.text = i.ToString();
            }
            else
            {
                countdownText.text = "START";
            }
            
            yield return new WaitForSeconds(1f);
        }

        countdownText.gameObject.SetActive(false);
        EventSidebar.EventSidebarEvents.StartExercises();
    }

    public static class GameManagerEvents
    {
        public static Action<Exercise> ChangeExercise;
        public static Action<string, float> CheckExercise;
        public static Action NextRepetition;
        public static Action ResetCurrRepAccuraty;
    }
}
