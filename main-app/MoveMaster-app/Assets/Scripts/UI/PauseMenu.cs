using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class PauseMenu : MonoBehaviour
{
    [SerializeField] private Button pauseButton;
    [SerializeField] private GameObject pausePanel;

    [SerializeField] private Button resumeButton;
    [SerializeField] private Button backToMenuButton;
    [SerializeField] private Button quitGameButton;

    [SerializeField] private TextMeshProUGUI countdownText;
    
    private PlayerInput _playerInput;
    private InputAction _pauseAction;
    
    private int _countdown = 5;
    private bool _isCountingDown = false;
    private bool _paused = false;
    private void Awake()
    {
        pauseButton.onClick.AddListener(PauseGame);
        resumeButton.onClick.AddListener(ResumeGame);
        backToMenuButton.onClick.AddListener(BackToMenu);
        quitGameButton.onClick.AddListener(QuitGame);

        _playerInput = gameObject.GetComponent<PlayerInput>();
        _pauseAction = _playerInput.actions["Pause"];
        _pauseAction.performed += ChangeGameState;
    }

    private void PauseGame()
    {
        if (_isCountingDown) return;
        _paused = true;
        pausePanel.SetActive(true);
        Time.timeScale = 0;
    }
    
    private void ResumeGame()
    {
        _paused = false;
        pausePanel.SetActive(false);
        countdownText.gameObject.SetActive(true);
        
        StartCoroutine(CountdownToResume());
    }

    private void BackToMenu()
    {
        SceneManager.LoadScene("MainMenu");
    }

    private IEnumerator CountdownToResume()
    {
        _isCountingDown = true;
        for (int i = _countdown; i >= 0; i--)
        {
            if (i != 0)
            {
                countdownText.text = i.ToString();
            }
            else
            {
                countdownText.text = "START";
            }
            
            yield return new WaitForSecondsRealtime(1f);
        }

        Time.timeScale = 1;
        countdownText.gameObject.SetActive(false);
        _isCountingDown = false;
    }
    private void QuitGame()
    {
        Application.Quit();
    }

    private void ChangeGameState(InputAction.CallbackContext context)
    {
        if (_paused)
        {
            ResumeGame();
        }
        else
        {
            PauseGame();
        }
    }
}
