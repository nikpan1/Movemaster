using System;
using TMPro;
using UnityEngine;

public class RepCounter : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI repCounterText;
    private float _howManyReps = 0;
    private int _repCounter = 0;

    private void OnEnable()
    {
        RepCounterEvents.IncreaseRepCounter += IncreaseRepCounter;
        RepCounterEvents.SetNewRepCounter += SetNewRepCounter;
    }

    private void OnDisable()
    {
        RepCounterEvents.IncreaseRepCounter -= IncreaseRepCounter;
        RepCounterEvents.SetNewRepCounter -= SetNewRepCounter;
    }


    private void IncreaseRepCounter() 
    {
        _repCounter++;
        repCounterText.text = _repCounter + "/" + _howManyReps;   
    }

    private void SetNewRepCounter(float howManyReps, string exerciseName)
    {
        if (exerciseName == "Idle")
        {
            repCounterText.gameObject.SetActive(false);
        }
        else
        {
            repCounterText.gameObject.SetActive(true);
            _repCounter = 0;
            _howManyReps = howManyReps;
            repCounterText.text = _repCounter + "/" + _howManyReps;
        }
    }

    public static class RepCounterEvents
    {
        public static Action IncreaseRepCounter;
        public static Action<float, string> SetNewRepCounter;
    }
}
