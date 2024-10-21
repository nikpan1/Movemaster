using System;
using UnityEngine;

public class RotablePlatform : MonoBehaviour
{
    [SerializeField] private float speed = 100;
    [SerializeField] private GameObject[] objectsOnPlatform;
    [SerializeField] private Camera camera;
   
    private bool _isRotating;
    private float _startMousePosition;
    private float _currentMousePosition;
    private float _mouseMovement;

    private void Start()
    {
        foreach (GameObject obj in objectsOnPlatform)
        {
            obj.transform.SetParent(gameObject.transform);
        }
    }

    private void Update()
    {
        // Idk which version is better
        //camera.fieldOfView += -Input.GetAxis("Mouse ScrollWheel") * 50000 * Time.deltaTime;
        camera.fieldOfView += -Input.GetAxis("Mouse ScrollWheel") * speed;
        
        if (Input.GetMouseButtonDown(0))
        {
            _isRotating = true;
            _startMousePosition = Input.mousePosition.x;
        }
        else if (Input.GetMouseButtonUp(0))
        {
            _isRotating = false;
        }

        if (_isRotating)
        {
            RotatePlatform();
        }
    }

    private void RotatePlatform()
    {
        _currentMousePosition = Input.mousePosition.x;
        _mouseMovement = _currentMousePosition - _startMousePosition;
        transform.Rotate(Vector3.up, -_mouseMovement * speed * Time.deltaTime);
        _startMousePosition = _currentMousePosition;
    }
}
