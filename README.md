
# Movemaster

## Table of Contents
- [About](#about)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Team](#team)

## About
**Movemaster** is a computer application designed to help users learn proper execution of basic strength and isometric exercises by offering an interactive and engaging gameplay experience. The application’s interface is simple and intuitive, making it easy for users of all skill levels to navigate. Player movements are tracked using computer vision, leveraging the camera of either a computer or smartphone. The gameplay is inspired by classic arcade games, where users earn points for performing exercises correctly. The accuracy of the movement will directly influence the score, providing feedback on the precision of each exercise.


## Project Structure
```
/project-root
│
├── /main-app
│   └── /MoveMaster-app          # Main Unity application project file
│
├── /mocker
│   └── /src			         # Python mocker app
│
├── /cv-server
│   └── /python-server         # Python server implementation
│
├── /mobile-dlc
│   └── /UnityMobileDLC        # Mobile application DLC for Unity
│
└── /scripts
    └── build.sh               # Script for building the project
    └── deploy.sh              # Script for deployment
```

## Getting Started

### Prerequisites
- Python 3.11
- Unity LTS version 2022.3.50f1

### Running the Application
- **Main Unity App:** Launch the Unity project and run the scene to start the game.
- **Server:** Navigate to the `/server` directory and run the Python server:
  ```bash
  python server.py
  ```

## Documentation
Detailed documentation is available in the `/docs` folder and `readme.md` of every subproject.

## Team
- Mathias Porzycki
- Nikodem Panknin
- Olga Maruszyńska
- Jakub Kwieciński
- Dawid Nowicki

Feel free to contact us if you have any questions or suggestions for the project.
```
