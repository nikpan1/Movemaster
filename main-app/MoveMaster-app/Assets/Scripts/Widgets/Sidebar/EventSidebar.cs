using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class EventSidebar : MonoBehaviour
{
    private class ExerciseItem
    {
        internal Exercise exercise;
        internal float repetitionCount;
        internal float durationTime;
        internal float repetitionTime;
    }
    
    [SerializeField] private GameObject previousExercise;
    [SerializeField] private GameObject currentExercise;
    [SerializeField] private GameObject nextExercise;
    [SerializeField] private ExercisesSet exercisesSet;
    [SerializeField] private GameObject endText;
    
    private Image _currentSprite;
    private Image _nextSprite;
    private Image _previousSprite;
    private Exercise _currentExercise;
    private List<ExerciseItem> _exerciseItems = new List<ExerciseItem>();

    private void Awake()
    {
        _previousSprite = previousExercise.GetComponent<Image>();
        _currentSprite = currentExercise.GetComponent<Image>();
        _nextSprite = nextExercise.GetComponent<Image>();
    }

    private void Start()
    {
        _previousSprite.enabled = false;
        if (exercisesSet != null)
        {
            SetExercisesItems();
        }

        StartCoroutine(ChangeExercise());
    }

    private void SetExercisesItems()
    {
        int multiplier = 1;
        float repTime;
        for (int i = 0; i < exercisesSet.ExercisesInSet.Count; i++)
        {
            if (exercisesSet.ExercisesInSet[i].TwoSideExercise) multiplier = 2;
            else multiplier = 1;
            
            for (int j = 0; j < exercisesSet.ExercisesInSet[i].HowManySeries*multiplier; j++)
            {
                repTime = exercisesSet.ExercisesInSet[i].ExerciseDuration /
                          exercisesSet.ExercisesInSet[i].HowManySeries;
                ExerciseItem exerciseItem = new ExerciseItem
                {
                    exercise = exercisesSet.ExercisesInSet[i].Exercise,
                    repetitionCount = exercisesSet.ExercisesInSet[i].RepetitionCount,
                    durationTime = repTime,
                    repetitionTime = repTime/exercisesSet.ExercisesInSet[i].RepetitionCount,
                };
                
                _exerciseItems.Add(exerciseItem);
            }
        }
    }

    private IEnumerator ChangeExercise()
    {
        _currentSprite.sprite = _exerciseItems[0].exercise.ExerciseSprite;
        _nextSprite.sprite = _exerciseItems[1].exercise.ExerciseSprite;
        GameManager.GameManagerEvents.ChangeExercise(_exerciseItems[0].exercise);
        
        StartCoroutine(Repetitions(_exerciseItems[0].repetitionCount, _exerciseItems[0].repetitionTime));
        yield return new WaitForSeconds(_exerciseItems[0].durationTime + 0.01f);
        
        _previousSprite.enabled = true;
        _previousSprite.sprite = _exerciseItems[0].exercise.ExerciseSprite;;

        for (int i = 1; i < _exerciseItems.Count; i++)
        {
            _currentSprite.sprite = _exerciseItems[i].exercise.ExerciseSprite;
            GameManager.GameManagerEvents.ChangeExercise(_exerciseItems[i].exercise);
            if (i + 1 > _exerciseItems.Count - 1)
            {
                _nextSprite.enabled = false;
                break;
            }
            
            _nextSprite.sprite = _exerciseItems[i + 1].exercise.ExerciseSprite;
            StartCoroutine(Repetitions(_exerciseItems[i].repetitionCount, _exerciseItems[i].repetitionTime));
            yield return new WaitForSeconds(_exerciseItems[i].durationTime + 0.01f);
            _previousSprite.sprite = _exerciseItems[i].exercise.ExerciseSprite;
        }

        yield return new WaitForSeconds(_exerciseItems[_exerciseItems.Count - 1].durationTime + 0.01f );
        _previousSprite.sprite = _exerciseItems[_exerciseItems.Count - 1].exercise.ExerciseSprite;
        _currentSprite.enabled = false;
        endText.SetActive(true);
    }

    private IEnumerator Repetitions(float repCount, float repTime)
    {
        GameManager.GameManagerEvents.ResetCurrRepAccuraty();
        for (int i = 0; i < repCount; i++)
        {
            yield return new WaitForSeconds(repTime);
            GameManager.GameManagerEvents.NextRepetition();
        }
    }
}