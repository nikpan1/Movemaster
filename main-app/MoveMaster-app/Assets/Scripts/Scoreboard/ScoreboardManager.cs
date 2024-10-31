using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class ScoreboardManager : MonoBehaviour
{
    public static ScoreboardManager Instance;
    private TMP_Text _scorePoints;
    private int _score=0;

    private void Awake(){
        if (Instance != null && Instance != this) { 
            Destroy(this); 
        } 
        else { 
            Instance = this; 
        } 
    }

    private void Start()
    {
        SetScore(0);
    }

    public void SetScore(int newScore){
        _score = newScore;
        _scorePoints.SetText(newScore.ToString());
    }

    public int GetScore(){
        return _score;
    }

    public void AddValue(int value){
        int newScore = _score + value;
        SetScore(newScore);
    }
}