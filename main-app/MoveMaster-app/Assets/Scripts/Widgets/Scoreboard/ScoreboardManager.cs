using System;
using TMPro;
using UnityEngine;

public class ScoreboardManager : MonoBehaviour
{
    public static ScoreboardManager Instance;
    private int _score;
    [SerializeField] private TMP_Text _scorePoints;

    [SerializeField]
    private MotivationPanel motivationPanel;
    
    private void OnEnable()
    {
        ScoreboardManagerEvents.AddValue += AddValue;
    }

    private void OnDisable()
    {
        ScoreboardManagerEvents.AddValue -= AddValue;
    }

    private void Awake()
    {
        if (Instance != null && Instance != this)
            Destroy(this);
        else
            Instance = this;
    }

    private void Start()
    {
        SetScore(0, 1);
    }

    public void SetScore(int newScore, float latestConfidence)
    {
        _score = newScore;
        _scorePoints.SetText(newScore.ToString());
        
        if (_score % 200 == 0 && _score > 0)
        {
            Debug.Log("Score: " + _score);
            motivationPanel.SetMotivationMessage(70, latestConfidence);
        }
        
    }

    public int GetScore()
    {
        return _score;
    }

    public void AddValue(int value, float latestConfidence)
    {
        var newScore = _score + value;
        SetScore(newScore, latestConfidence);
    }

    public static class ScoreboardManagerEvents
    {
        public static Action<int, float> AddValue;
    }
}