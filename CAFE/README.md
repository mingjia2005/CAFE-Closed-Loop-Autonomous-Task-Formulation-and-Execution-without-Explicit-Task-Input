
# aithor2-v5 Project Guide

## Directory Overview

- `aithor2/`
  - `main_with_depth.py`: Main entry point, responsible for overall workflow control, invoking lightweight LLM understanding and navigation.
  - `requirements.txt`: Main dependency list.
  - `autonomous_navigation.py`, `frontier_fullmap_navigation.py`, `pointnav_navigator.py`, etc.: Navigation and exploration modules.
  - `display_manager.py`, `topdown_camera_manager.py`, `topdown_ui_renderer.py`, `topdown_view.py`, `official_topdown_view.py`: Visualization and UI modules.
  - `exploration_io.py`, `explore_topdown_data.py`, `semantic_priors.py`, `object_norms.py`, `storage_scoring.py`, `structured_export.py`: Exploration, semantics, scoring, and export modules.
  - `scene_state_manager.py`, `input_handler.py`, `io_interfaces.py`, `known_map_navigator.py`, `viewpoint_navigation.py`: Scene state, I/O, known map navigation, etc.
  - `yolo_utils.py`, `yolov8n.pt`: YOLO object detection code and model.
  - `lightweight_llm_monitor.py`: Lightweight LLM monitoring module.
  - `embodied B1/`: Core directory for lightweight LLM, see below for details.

- `semantic_maps/`
  - `current_scene_state.json`, etc.: Store scene state, preferences, exploration candidates, and other data files.

## Lightweight LLM Related (`aithor2/embodied B1/`)

- `main.py`: Main logic entry for lightweight LLM, dynamically loaded and called by `main_with_depth.py`.
- `config.py`: Configuration file, stores API keys, model parameters, etc. (Do not leak your keys).
- `prompts.py`, `prompts_new.py`: Store LLM prompts, imported by the main program.
- `api.py`, `mab.py`, `world.py`, etc.: Implement LLM API wrappers, multi-armed bandit algorithms, world models, etc.
- `requirements.txt`: Dependency list for this submodule.

## Main Function

- The main entry is `aithor2/main_with_depth.py`, responsible for overall scheduling.
- This main program dynamically loads `aithor2/embodied B1/main.py` and calls its core classes such as `ThreeLLMSystemV2` for understanding and task planning.
- Prompts and configuration for the lightweight LLM are in `aithor2/embodied B1/prompts.py` and `aithor2/embodied B1/config.py`.

## Environment Setup

- Install main dependencies:
  ```bash
  pip install -r aithor2/requirements.txt
  ```
- Install lightweight LLM submodule dependencies:
  ```bash
  pip install -r aithor2/embodied B1/requirements.txt
  ```

## Other Notes

- `.venv` and `.venv_ai2thor` are virtual environment folders. Keep or remove them as needed.
- Do not expose your personal API keys in config.py or other files.

---
For more detailed module descriptions or usage instructions, please provide additional requirements.
