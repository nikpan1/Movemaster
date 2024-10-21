# Project Structure

The project has an organized folder structure to facilitate navigation and resource management. Each type of asset (graphics, audio, scripts) is properly categorized based on the element it relates to. This way, we can quickly find the necessary files without searching through a clutter of unrelated assets.

## Folder Structure:

```
Assets/
├── Graphics/
│   ├── 2D/
│   │   ├── UI/
│   │   └── Temporary/
│   └── 3D/
│       ├── SomExampleModule/
│       └── Environment/
├── Prefabs/
│   ├── SomExampleModule/
│   └── Environment/
├── Scripts/
│   ├── SomExampleModule/
│   ├── Environment/
│   └── Scriptable Objects/
├── Scriptable Objects/
│   └── Exercises/
└── Audio/
    ├── SomExampleModule/
    └── Environment/
```

### Description of Each Directory:

- **Graphics/2D**
  - `UI/` - User interface elements (buttons, panels, etc.).
  - `Temporary/` - Temporary graphics, e.g., exercise poses.

- **Graphics/3D**
  - `SomExampleModule/` - 3D models and materials related to SomExampleModule.
  - `Environment/` - 3D elements of the game's environment.

- **Prefabs**
  - `SomExampleModule/` - Prefabs related to SomExampleModule.
  - `Environment/` - Environment prefabs.

- **Scripts**
  - `SomExampleModule/` - Scripts that manage SomExampleModule.
  - `Environment/` - Scripts related to the environment.
  - `Scriptable Objects/` - Scripts for Scriptable Objects.

- **Scriptable Objects**
  - `Exercises/` - Scriptable Objects defining exercises.

- **Audio**
  - `SomExampleModule/` - Sounds related to SomExampleModule.
  - `Environment/` - Environment sounds.

Each folder contains resources related to specific elements of the project, making it easier to manage and expand the game as the project grows.