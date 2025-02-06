


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/nikpan1/Movemaster">
    <!-- You may place a project logo or relevant image here -->
    <img src="main-app/MoveMaster-app/Assets/Graphics/2D/GUI/logo.png" alt="Logo">
  </a>

  <h3 align="center">Movemaster</h3>

  <p align="center">
    A computer vision-based application for learning and practicing strength exercises through engaging gameplay.
    <br />
    <a href="https://github.com/nikpan1/Movemaster"><strong>Explore the documentation »</strong></a>
    <br />
    <br />
    <a href="https://github.com/nikpan1/Movemaster">View Demo</a>
    ·
    <a href="https://github.com/nikpan1/Movemaster/issues/new?labels=bug">Report Bug</a>
    ·
    <a href="https://github.com/nikpan1/Movemaster/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

 -  -  -  - 

<!-- ABOUT THE PROJECT -->
## About the Project

Movemaster is a computer application designed to assist users in learning and performing basic strength and isometric exercises through interactive gameplay. It leverages computer vision to track the user’s movements from a standard camera (e.g., a laptop webcam or phone camera) and assesses the precision of each exercise in real time.

The core gameplay mechanics are inspired by classic arcade systems (e.g., Beat Saber, Dance Central) and reward players with points based on accurate movements. However, unlike those games, Movemaster does not require specialized hardware such as VR goggles or Kinect sensors, making it accessible to a broader audience.

Key project highlights:
* **User-friendly**: Operates with minimal hardware requirements (standard camera).
* **Engaging**: Incorporates arcade elements to sustain user motivation.
* **Modular**: Combines Unity 3D with a Python-based movement analysis module.
* **Scalable**: Uses websockets to communicate between Unity and the Python backend, allowing for flexible system architecture.

This project follows an agile (Scrum) methodology. Tools such as Git for version control and Taiga for task management ensure efficient collaboration and code quality. Development progresses iteratively, from initial market research and technology feasibility to final implementation of a fully functional application.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

 -  -  -  - 

## Project Structure
```
/project-root
│
├── /main-app
│   └── /MoveMaster-app          # Main Unity application project file
│
├── /cv-server                   # Computer Vision python server
│
├── /model                       # AI model implementation
│
└── ...
```

 -  -  -  - 

# Contributors
We extend our gratitude to everyone who has contributed to the development and improvement of Movemaster. Your feedback, pull requests, and suggestions make this project stronger.

<a href="https://github.com/nikpan1/Movemaster/graphs/contributors"> <img src="https://contrib.rocks/image?repo=nikpan1/Movemaster" alt="Contributors" /> </a></br>
<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- CONTACT -->

 -  -  -  - 

# Contact
Project Maintainer</br>
nikpan1 – nikodem.panknin@gmail.com</br>
https://github.com/nikpan1/Movemaster

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- ACKNOWLEDGMENTS -->


# Workspace
The project on the taiga that was used during development is now available to everyone, take a look below:</br>
https://tree.taiga.io/project/nikpan1-gruz-kasztany-i-szarlotki
