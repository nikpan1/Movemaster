using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using Random = UnityEngine.Random;

public class MotivationPanel : MonoBehaviour
{
    [Serializable]
    class TextVariant
    {
        [SerializeField] private string text;
        [SerializeField] private Color color;

        public string Text => text;
        public Color Color => color;
    }

    [SerializeField] private TextMeshProUGUI motivationText;
    [SerializeField] private RectTransform textArea;
    [SerializeField] private List<TextVariant> textVariants;
    [SerializeField] private float pulseSpeed = 1f;
    [SerializeField] private float pulseScale = 1.1f;

    [SerializeField] private float maxRotationAngle = 30f;
    bool isAnimating = false;
    
    private void OnEnable()
    { 
        MotivationPanelEvents.SetMotivationMessage += SetMotivationMessage;
    }

    private void OnDisable()
    {
        MotivationPanelEvents.SetMotivationMessage -= SetMotivationMessage;
    }

    public void SetMotivationMessage(int score, float latestConfidence)
    {
        if (isAnimating) return;
        int randomIndex = RandomIndex(score);
        SetTextAndColor(randomIndex);  
        
        motivationText.rectTransform.rotation = Quaternion.Euler(0, 0, UnityEngine.Random.Range(-maxRotationAngle, maxRotationAngle));
        motivationText.gameObject.SetActive(true);
        isAnimating = true;
        StartCoroutine(PulseEffect());
        StartCoroutine(HideMotivationMessage());
    }

    private void SetTextAndColor(int index)
    {
        motivationText.text = textVariants[index].Text;
        motivationText.color = textVariants[index].Color;
    }
    
    private IEnumerator HideMotivationMessage()
    {
        yield return new WaitForSecondsRealtime(2f);
        motivationText.gameObject.SetActive(false);
        isAnimating = false;
    }
    
    private IEnumerator PulseEffect()
    {
        while (motivationText.gameObject.activeSelf)
        {
            float scaleFactor = Mathf.Sin(Time.time * pulseSpeed) * (pulseScale - 1) + 1;
            motivationText.rectTransform.localScale = new Vector3(scaleFactor, scaleFactor, 1);
            yield return null;
        }
    }
    
    private int RandomIndex(int score)
    {
        return Random.Range(0, textVariants.Count);
    }

    public static class MotivationPanelEvents
    {
        public static Action<int, float> SetMotivationMessage;
    }
}