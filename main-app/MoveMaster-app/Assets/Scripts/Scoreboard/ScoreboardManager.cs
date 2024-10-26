using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class ScoreboardManager : MonoBehaviour
{
    public static ScoreboardManager Instance;
    [SerializeField]
    private TMP_Text scorePoints;
    [SerializeField]
    private int score=0;

    void Awake(){
        if (Instance != null && Instance != this) 
    { 
        Destroy(this); 
    } 
    else 
    { 
        Instance = this; 
    } 
    }

    void Start()
    {
        setScore(0);
    }

    public void setScore(int newScore){
        score = newScore;
        scorePoints.SetText(newScore.ToString());
    }

    public int getScore(){
        return score;
    }

    public void addValue(int value){
        int newScore = score + value;
        setScore(newScore);
    }
}
