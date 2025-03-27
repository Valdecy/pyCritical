# pyCritical

## Introduction

**pyCritical** is a Python library for solving **project scheduling problems** using the **Critical Path Method (CPM)** and the **Program Evaluation and Review Technique (PERT)**. It offers computational tools and visualizations to help identify the critical path, compute task timings, and assess project duration risks.

Whether you're planning a construction project, coordinating a research effort, or teaching operations research, `pyCritical` helps you model, analyze, and visualize complex project schedules.

---

## Features

- **CPM Analysis**  
  - Earliest Start (ES), Earliest Finish (EF)
  - Latest Start   (LS), Latest Finish   (LF)
  - Slack time for each task
  - Full **critical path** identification

- **PERT Analysis**  
  - Support for **three-point estimates** (Optimistic, Most Likely, Pessimistic)
  - Computes expected project duration and standard deviation
  - Probability-based assessment of project completion
  - Earliest Start (ES), Earliest Finish (EF)
  - Latest Start   (LS), Latest Finish   (LF)
  - Slack time for each task
  - Full **critical path** identification

- **Support for Lags and Dependency Types**  
  - `Finish-to-Start  (FS): Task B can't start  until Task A finishes.`
  - `Start-to-Start   (SS): Task B can't start  until Task A starts.`
  - `Finish-to-Finish (FF): Task B can't finish until Task A finishes.`
  - `Start-to-Finish  (SF): Task B can't finish until Task A starts.`  
  - Including **positive or negative lag times** between tasks.

- **Gantt Chart Visualization**  
  - Clear, professional Gantt charts with critical paths highlighted
  - Dependency arrows with labels (e.g., `FS+2`)
  - Optional display of task slack

---

## Installation

```bash
pip install pycritical
```

---

## Usage

Try it in **Colab**:

- Example 01: CPM  ([ Colab Demo ](https://colab.research.google.com/drive/1d9Hrldzh5qnSQlYUhjmsiHh6Tv6G3CF5?usp=sharing))
- Example 02: PERT ([ Colab Demo ](https://colab.research.google.com/drive/1RQt0MSD6j7GPT6_K3_8gqaSGPgflh6U5?usp=sharing))
- Example 03: CPM  with Dep./Lags ([ Colab Demo ](https://colab.research.google.com/drive/1Kh_E4U_KPWxrvdWsfcW9jRdxAF-lAvYR?usp=sharing))
- Example 04: PERT with Dep./Lags ([ Colab Demo ](https://colab.research.google.com/drive/1Y-uuIZcAP7b_qJddV93K5Vp8Eaeh4uhp?usp=sharing))

---
