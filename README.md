## Paper Title

**"I understand why I got this grade": Automatic Short Answer Grading (ASAG) with Feedback**

## Authors

Dishank Aggarwal, Pritam Sil, Bhaskaran Raman, Pushpak Bhattacharyya

## Abstract

In recent years, there has been a growing interest in using Artificial Intelligence (AI) to automate student assessment in education. Among different types of assessments, summative assessments play a crucial role in evaluating a student's understanding level of a course. Such examinations often involve short-answer questions. However, grading these responses and providing meaningful feedback manually at scale is both time-consuming and labor-intensive. Feedback is particularly important, as it helps students recognize their strengths and areas for improvement. Despite the importance of this task, there is a significant lack of publicly available datasets that support automatic short-answer grading with feedback generation. To address this gap, we introduce Engineering Short Answer Feedback (EngSAF), a dataset designed for automatic short-answer grading with feedback. Clef, containing ~5.8k data points. We incorporate feedback into our dataset by leveraging the generative capabilities of state-of-the-art large language models (LLMs) using our Label-Aware Synthetic Feedback Generation (LASFG) strategy. This paper underscores the importance of enhanced feedback in practical educational settings, outlines dataset annotation and feedback generation processes, conducts a thorough EngSAF analysis, and provides different LLMs-based zero-shot and finetuned baselines for future comparison. The best-performing model (Mistral-7B) achieves an overall accuracy of 75.4% and 58.7% on unseen answers and unseen question test sets, respectively. Additionally, we demonstrate the efficiency of our ASAG systems through its deployment in a real-world end-semester exam, achieving an output label accuracy of 92.5% along with feedback quality and emotional impact scores of 4.5 and 4.9 (out of 5) on human evaluation, thus showcasing its practical viability and potential for broader implementation in educational institutions.

## Dataset Details
- **EngSAF Dataset**: Contains ~5.8k data points from real-life examinations at a reputed institute, covering 119 questions across multiple engineering domains.
- **Data Split**: 70% training, 16% unseen answers (UA), 14% unseen questions (UQ).

## EngSAF Dataset Access
The dataset for "**I understand why I got this grade": Automatic Short Answer Grading (ASAG) with Feedback** is available upon request. To request access, please complete the [Dataset Access Request Form](https://forms.gle/TYDydJAq65imFsLJ6). After review, approved requesters will receive a secure link to download the dataset.

**Terms of Use**:
- The dataset may only be used for the purpose specified in the request.
- Redistribution of the dataset is prohibited.
- Any publications using the dataset must cite:

## Citation
If you use this work in your research, please cite our paper:
```
Aggarwal, D., Sil, P., Raman, B., & Bhattacharyya, P. (2025). "I understand why I got this grade": Automatic Short Answer Grading (ASAG) with Feedback. To appear in Proceedings of the 26th International Conference on Artificial Intelligence in Education (AIED 2025).
```

## Contact
For questions or issues, please contact dishank.aggarwal@cse.iitb.ac.in or open an issue in this repository.
