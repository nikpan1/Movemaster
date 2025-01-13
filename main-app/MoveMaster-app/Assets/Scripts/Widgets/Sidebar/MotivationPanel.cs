using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.Windows;

public class MotivationPanel : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI motivationText;
    private String[] _motivationsStrings = {"Perfect", "Excellent", "Great", "Good", "You can do it better"};
    [SerializeField] private Color[] _motivationsColors = new Color[5];

    private void OnEnable()
    {
        MotivationPanelEvents.SetMotivationMessage += SetMotivationMessage;
    }

    private void OnDisable()
    {
        MotivationPanelEvents.SetMotivationMessage -= SetMotivationMessage;
    }

    private void SetMotivationMessage(int score, bool isBreakdown)
    {
        if (isBreakdown) return;
        if (score <= 100 && score >= 90) SetTextAndColor(0);
        else if (score < 90 && score >= 75) SetTextAndColor(1);
        else if (score < 75 && score >= 50) SetTextAndColor(2);
        else if (score < 50 && score >= 25) SetTextAndColor(3);
        else SetTextAndColor(4);
        motivationText.gameObject.SetActive(true);
        StartCoroutine(HideMotivationMessage());
    }

    private void SetTextAndColor(int index)
    {
        motivationText.text = _motivationsStrings[index];
        motivationText.color = _motivationsColors[index];
    }
    
    private IEnumerator HideMotivationMessage()
    {
        yield return new WaitForSecondsRealtime(3f);
        motivationText.gameObject.SetActive(false);
    }

    public static class MotivationPanelEvents
    {
        public static Action<int, bool> SetMotivationMessage;
    }
}
