using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Stickman : MonoBehaviour
{
    [SerializeField] private GameObject stickman;
    private Animator _stickmanAnimator;

    private void Awake()
    {
        _stickmanAnimator = stickman.GetComponent<Animator>();
    }

    private void OnEnable()
    {
        StickmanEvents.PlayAnimation += PlayAnimation;
    }

    private void OnDisable()
    {
        StickmanEvents.PlayAnimation -= PlayAnimation;
    }
    private void PlayAnimation(string animationName)
    {
        //AnimatorStateInfo stateInfo = _stickmanAnimator.GetCurrentAnimatorStateInfo(0);
        //if(stateInfo.IsName(animationName)) _stickmanAnimator.Play(animationName, -1, 0f);
        //else _stickmanAnimator.CrossFade(animationName, 0.2f);
        
        _stickmanAnimator.Play(animationName, -1, 0f);
    }

    public static class StickmanEvents
    {
        public static Action<string> PlayAnimation;
    }
}
