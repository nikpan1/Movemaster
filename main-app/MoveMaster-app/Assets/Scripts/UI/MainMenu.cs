using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class MainMenu : MonoBehaviour
{
    [SerializeField] private Button startButton;
    [SerializeField] private Button levelsButton;
    [SerializeField] private Button settingsButton;
    [SerializeField] private Button creditsButton;
    [SerializeField] private Button backButton;
    [SerializeField] private Button quitButton;

    [SerializeField] private GameObject menuPanel;
    [SerializeField] private GameObject levelsPanel;
    [SerializeField] private GameObject settingsPanel;
    [SerializeField] private GameObject creditsPanel;

    private List<GameObject> _panels;
    private void Awake()
    {
        _panels =  new List<GameObject>(){
            menuPanel,
            levelsPanel,
            settingsPanel,
            creditsPanel
        };
        startButton.onClick.AddListener(StartGame);
        levelsButton.onClick.AddListener(() => { ShowPanel(levelsPanel); });
        settingsButton.onClick.AddListener(() => { ShowPanel(settingsPanel); });
        creditsButton.onClick.AddListener(() => { ShowPanel(creditsPanel); });
        backButton.onClick.AddListener(() => { ShowPanel(menuPanel); });
        quitButton.onClick.AddListener(CloseApp);
    }

    private void StartGame()
    {
        SceneManager.LoadScene("MainScene");
    }

    private void ShowPanel(GameObject panelToShow)
    {
        foreach (GameObject panel in _panels) 
        {
            if (panel == panelToShow)
            {
                panel.SetActive(true);
            }
            else
            {
                panel.SetActive(false);
            }
        }
        if(panelToShow == menuPanel) backButton.gameObject.SetActive(false);
        else backButton.gameObject.SetActive(true);
    }

    private void CloseApp()
    {
        Application.Quit();
    }
}
