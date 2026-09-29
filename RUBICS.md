# CITS4403 Research Project

## Technical Specifications and Assessment Rubric

**18 September 2026**

This document provides the technical specifications and assessment rubrics for the research project.

---

## 1 Technical Specification

For all project development and code implementation, you may use any tools, libraries, or utilities. However, your final project must be developed, implemented, and runnable in **Python 3.x**. External tools or languages may be used to generate intermediate results, but the final deliverable must execute in Python 3.x.

Each team must collaborate using GitHub. Create a private shared repository where both members contribute. The repository must be well-structured as shown below. You will later be asked to make the repository public for assessment. If you are not familiar with using GitHub, you can refer to the GitHub workshop. Additional help will also be available during the lab sessions.

```
project-root/
|
+-- src/               % All main code (functions, classes, models, etc.)
+-- utils/             % Utility/helper functions
+-- data/              % Datasets or data samples (if applicable)
+-- notebooks/         % Jupyter Notebooks (analysis, results, demonstrations)
+-- requirements.txt   % List of dependencies
+-- README.md          % Project overview, setup instructions, usage guide
```

You may not directly copy code from open-source projects on GitHub or other platforms. Replicating or extending existing work is allowed, but you must clearly state your own contributions and modifications. In addition, you may not analyse cases already covered in lectures or in the textbook, including Game of Life, Schelling's model of segregation, Sugarscape model of wealth distribution, traffic jams, bird flocks, evolution, or the Prisoner's Dilemma. If your work builds on these models (covered in lectures or the unit textbook), you must explicitly highlight your novel contribution or extension.

> **Important:** Any submission found to violate any of these rules will receive **zero marks** for this project.

All projects must comply with the restrictions above. The Originality and Contribution criterion assesses the value of the team's contribution within projects that meet these requirements. Limited originality may reduce the mark for this criterion, but does not, by itself, result in zero marks for the whole project.

---

## 2 Assessment

The project contributes **25%** to the unit mark, comprising project code (10%), report (10%), and demonstration (5%). Assessment considers both group work and individual contributions, as indicated in the rubric.

### 2.1 Project Code Rubric (10%)

The code component is assessed using the submitted implementation, supporting notebooks, GitHub activity, collaboration and responses to implementation questions.

Each criterion is assessed at one of four levels: **Advanced**, **Proficient**, **Approaching proficiency**, or **Beginning**.

| Criterion | What is assessed |
|---|---|
| **Execution and Completeness** | • The project runs in Python 3.x using the documented setup and usage instructions. • All code used to produce the reported results is supplied, together with the required dependencies and any necessary data or data-access instructions. |
| **Model Correctness and Testing** | • Formulae, algorithms and model rules are implemented correctly, with appropriate control flow and data structures. • Appropriate checks or case analyses are used to establish correct model behaviour, including boundary or exceptional cases where relevant. |
| **Clarity and Modularity** | • Uses consistent formatting, meaningful names and documentation for major functions, variables and non-trivial algorithms. • Decomposes the problem into reusable functions or classes and avoids unnecessary repetition. |
| **Use of Jupyter Notebook as a Communication Tool** | • Blends code, equations, explanatory text, visualisations and relevant multimedia into a coherent account of the analysis. • The notebook sequence and any interactive elements are easy to follow and use. |
| **Effective Use of GitHub** *(individual)* | • Makes regular, meaningful commits with clear messages. • Uses issues to track bugs, tasks or missing features. • Regular use of GitHub pull requests with descriptive titles to add new features and fix bugs. |
| **Teamwork & Collaboration** *(individual)* | • Contributes constructively to shared development and discussions on a partner's issues. • Provides helpful code reviews and responds to feedback before merging changes. |
| **Code Understanding and Explanation** *(individual)* | • Explains their own contributions in detail and how these connect to the overall implementation. • Accurately explains how key functions, variables and control flow implement the model's rules. • Justifies relevant implementation choices and answers targeted questions about code behaviour, including the likely effect of a proposed change. |

---

### 2.2 Project Report (10%)

The report should be a maximum of **five A4 pages**, excluding figures, references and appendices, and should use an **11pt font with 1-inch margins** on all sides. Include links to relevant simulations where applicable.

**Scientific-report requirement.** Write the report in coherent paragraphs, using a logical structure, precise terminology and appropriate referencing. Bullet points may be used for brief lists, but must not replace explanations, justification of modelling choices or interpretation of results. Reports consisting mainly of bullet-point notes, screenshots or code descriptions, without sufficient explanation and analysis, are unlikely to achieve a high overall report mark.

Each criterion is assessed at one of four levels: **Advanced**, **Proficient**, **Approaching proficiency**, or **Beginning**.

| Criterion | What is assessed |
|---|---|
| **Background and Research Aims** | • Establishes the problem, its significance and a focused research question or modelling objective. • Summarises relevant studies and places the project in the context of existing literature. |
| **Originality and Contribution** | • Clearly distinguishes the team's contribution from existing approaches or taught examples. • Explains the purpose of that contribution and demonstrates its added value or insight. |
| **Model Specification and Justification** | • Defines the components, variables, interactions, equations or update rules needed to understand the model. • States assumptions and initialisation rules, and justifies relevant model-design and parameter choices. |
| **Experimental Design** | • Uses appropriate experiments or case studies to investigate the research question and explore model behaviour. • Specifies scenarios, parameter ranges, outcome measures and, where relevant, repeated runs or comparisons. • Provides sufficient procedural detail to repeat the experiments. |
| **Results** | • Reports relevant qualitative and quantitative findings accurately and provides sufficient evidence of model behaviour under the investigated conditions. • Uses well-chosen figures and tables with informative captions, labels and units where applicable. |
| **Discussion and Conclusions** | • Interprets findings in relation to the research question, rather than merely repeating the results. • Draws evidence-supported conclusions and explains the insights gained about the system. • Identifies limitations, uncertainty and potential improvements or applications without overstating the model's scope. |
| **Scientific Writing and Presentation** | • Uses coherent paragraphs, clear academic language and a logical report structure. • Uses terminology precisely and writes concisely, with grammatically clear expression. • Uses consistent citations and references, and follows the stated page, font and margin requirements. |

---

### 2.3 Project Demonstration (5%)

Please follow the instructions in the sign-up sheet to select a presentation slot. Plan your presentation in advance so that both team members participate. Ensure that you are both present five minutes before your scheduled time, with the model ready to run on a computer and all required data files and outputs available.

Your demonstration should take no more than **10 minutes** and should:

- Explain your motivation, provide a brief background, and state the project aims.
- Describe your modelling approach and justify your key choices.
- Present your results using relevant figures, plots, animations or simulations.
- Discuss your conclusions and the insights your model provides about the system.

After your demonstration, the marker will ask questions to clarify aspects of your work. Both team members should be prepared to respond, and their answers will be assessed individually.
