using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MakeExercise : MonoBehaviour
{
    [SerializeField] private Vector3[] targetPositions;
    [SerializeField] private Transform[] targetObjects;
    [SerializeField] private float moveSpeed = 1.0f;

    private Vector3[] _startPositions;

    void Start()
    {
        _startPositions = new Vector3[targetObjects.Length];
        for (int i = 0; i < targetObjects.Length; i++)
        {
            _startPositions[i] = targetObjects[i].position;
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            for (int i = 0; i < targetObjects.Length; i++)
            {
                print(targetPositions[i]);
                var vector3 = targetObjects[i].position;
                vector3.y = targetPositions[i].y;
                targetObjects[i].position = vector3;
                //vector3.x = targetPositions[i].x;
                //targetObjects[i].position = vector3;
            }
        }
    }
}
