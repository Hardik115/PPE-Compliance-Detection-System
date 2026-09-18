import os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def main():
    doc = Document()
    
    # Title
    title = doc.add_heading('Project Report: Mask & PPE Compliance Detection System', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # 1. Cover Page
    doc.add_heading('1. Cover Page', level=1)
    doc.add_paragraph('Project Title: Automated Face Mask & PPE Compliance Detection System')
    doc.add_paragraph('Course: Computer Vision CSE3010')
    doc.add_paragraph('Student Name: Hardik Gaur')
    doc.add_paragraph('Registration Number: 24BAI10484')
    doc.add_paragraph('Date: September 2026')
    doc.add_paragraph('GitHub Repository: https://github.com/Hardik115/PPE-Compliance-Detection-System')
    
    # 2. Introduction
    doc.add_heading('2. Introduction', level=1)
    doc.add_paragraph('In high-risk environments such as hospitals and manufacturing plants, compliance with Personal Protective Equipment (PPE) guidelines—specifically face masks—is critical. Manual monitoring is labour-intensive and prone to human error. This project introduces a fully automated, computer-vision-based system to detect faces in real-time, classify their mask compliance status, and log violations. Designed specifically to run efficiently on standard CPU hardware, the system provides a scalable, cost-effective solution for safety officers.')
    
    # 3. Problem Statement
    doc.add_heading('3. Problem Statement', level=1)
    doc.add_paragraph('How can we automate real-time face-mask compliance monitoring across multiple camera feeds on commodity CPU hardware, without requiring specialist GPU infrastructure, while generating a full audit trail of violations and on-demand compliance reports?')
    
    # 4. Functional Requirements
    doc.add_heading('4. Functional Requirements', level=1)
    doc.add_paragraph('1. Face Detection: Detect and localise human faces in real-time from webcam/video feeds.\n'
                      '2. Mask Classification: Classify each detected face as "Mask-Compliant" or "Non-Compliant" with a confidence score.\n'
                      '3. Logging & Storage: Persist violation events (timestamp, confidence, face snapshot, camera ID) into an SQLite database.\n'
                      '4. Analytics Reporting: Generate dynamic HTML compliance reports featuring statistical charts (pie, bar, histogram).')
    
    # 5. Non-Functional Requirements
    doc.add_heading('5. Non-Functional Requirements', level=1)
    doc.add_paragraph('1. Performance: Inference time must be <= 200 ms per frame on a standard CPU.\n'
                      '2. Accuracy: The classifier must achieve >= 95% accuracy on the test dataset.\n'
                      '3. Maintainability: The codebase must be highly modular (separate logic for detection, classification, logging, and reporting).\n'
                      '4. Usability: The system must be fully executable via the command-line interface without requiring a graphical setup wizard.')
    
    # 6. System Architecture
    doc.add_heading('6. System Architecture', level=1)
    doc.add_paragraph('The system follows a linear pipeline architecture:\n'
                      '1. Input Stage: OpenCV captures frames from a video file or live webcam.\n'
                      '2. Module 1 (Detector): The frame is passed through a Haar Cascade (or MediaPipe) detector to extract face bounding boxes.\n'
                      '3. Module 2 (Classifier): Each face crop is passed to a MobileNetV2 Convolutional Neural Network (CNN) to predict mask compliance.\n'
                      '4. Module 3 (Logger): Non-compliant events are securely written to an SQLite database using thread-safe context managers.\n'
                      '5. Module 4 (Report Generator): A separate module queries the database to build an HTML compliance dashboard.')
    
    # 7. Design Diagrams
    doc.add_heading('7. Design Diagrams', level=1)
    doc.add_paragraph('Note: The actual high-resolution images are located in the docs/ folder of the repository.')
    doc.add_paragraph('- System Architecture Diagram: docs/architecture_diagram.png\n'
                      '- UML Use Case Diagram: docs/uml_use_case.png\n'
                      '- UML Class Diagram: docs/uml_class.png\n'
                      '- UML Sequence Diagram: docs/uml_sequence.png\n'
                      '- ER Diagram: docs/er_diagram.png')
    
    # 8. Design Decisions & Rationale
    doc.add_heading('8. Design Decisions & Rationale', level=1)
    doc.add_paragraph('- MobileNetV2 over ResNet/VGG: Selected for its inverted residual blocks and lightweight profile. This guarantees our core requirement: real-time CPU-only inference.\n'
                      '- SQLite over PostgreSQL/MySQL: Chosen because it is serverless and embedded. This reduces the installation footprint to zero, adhering to our usability NFR.\n'
                      '- Haar Cascades over Deep Learning Detectors (YOLO): Selected for the initial face-cropping stage because Haar Cascades are computationally cheap and extremely fast on CPUs, leaving more processing power for the CNN classifier.')
    
    # 9. Implementation Details
    doc.add_heading('9. Implementation Details', level=1)
    doc.add_paragraph('The system is built in Python 3.9+ using OpenCV and TensorFlow/Keras.\n'
                      '- Training (train.py): Implements a two-phase transfer learning approach. First, the classification head is trained while the MobileNetV2 base is frozen. Then, the top 20 layers are fine-tuned.\n'
                      '- Detector (detector.py): Uses the Factory pattern to allow switching between OpenCV Haar Cascades and MediaPipe.\n'
                      '- Logger (logger.py): Utilizes threading.Lock() to ensure database integrity when handling rapid, multi-threaded video frames.')
    
    # 10. Screenshots / Results
    doc.add_heading('10. Screenshots / Results', level=1)
    doc.add_paragraph('- Live HUD: The terminal and OpenCV window display real-time FPS, total faces, violation counts, and bounding boxes (Green = Mask, Red = No Mask).\n'
                      '- HTML Report: Generates a timestamped report containing a donut chart of compliance rates, an hourly bar chart of violations, and a table of the 50 most recent events.')
    
    # 11. Testing Approach
    doc.add_heading('11. Testing Approach', level=1)
    doc.add_paragraph('The project follows a rigorous Test-Driven Development (TDD) approach using pytest.\n'
                      '- 66 Unit Tests: Covering 100% of the core business logic.\n'
                      '- Mocking: The database is tested using an in-memory SQLite instance (:memory:) to prevent polluting the actual logs during test runs. The TensorFlow model is mocked during testing to ensure fast test execution.\n'
                      '- Validation: The model was evaluated on a held-out 20% test split, achieving over 99% accuracy during the fine-tuning phase.')
    
    # 12. Challenges Faced
    doc.add_heading('12. Challenges Faced', level=1)
    doc.add_paragraph('- CPU Bottlenecks: Initial iterations using heavier models caused extreme lag (2-3 FPS). Moving to MobileNetV2 and applying frame-skipping techniques resolved this, bringing performance up to real-time standards.\n'
                      '- Database Concurrency: Rapidly logging frames caused sqlite3.OperationalError: database is locked. This was solved by implementing a custom context manager with threading locks.\n'
                      '- Cross-Platform Encoding: Windows terminal encoding issues caused crashes when printing unicode characters during training. This was resolved by sanitising console outputs.')
    
    # 13. Learnings & Key Takeaways
    doc.add_heading('13. Learnings & Key Takeaways', level=1)
    doc.add_paragraph('- Gained practical experience in Transfer Learning and fine-tuning pre-trained ImageNet models for custom binary classification.\n'
                      '- Learned how to properly structure a Python project for modularity, separating the UI (orchestrator) from the business logic (database, ML models).\n'
                      '- Developed a deep understanding of software testing, specifically how to mock hardware-dependent components (like webcams and ML models) to write reliable unit tests.')
    
    # 14. Future Enhancements
    doc.add_heading('14. Future Enhancements', level=1)
    doc.add_paragraph('- Multi-Camera Support: Extending the orchestrator to handle RTSP streams from multiple IP cameras simultaneously.\n'
                      '- Real-Time Alerts: Integrating an SMTP module to send email/SMS alerts to safety officers when a violation occurs.\n'
                      '- Facial Recognition (Re-ID): Identifying who the non-compliant person is by matching them against an employee database.')
    
    # 15. References
    doc.add_heading('15. References', level=1)
    doc.add_paragraph('1. OpenCV Documentation: https://docs.opencv.org/\n'
                      '2. MobileNetV2 Paper (Sandler et al., 2018): https://arxiv.org/abs/1801.04381\n'
                      '3. SQLite Python Documentation: https://docs.python.org/3/library/sqlite3.html')
                      
    doc.save('docs/Project_Report.docx')
    print("Word document generated at docs/Project_Report.docx")

if __name__ == '__main__':
    main()
