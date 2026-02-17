# ⚡ Smart Meter Chatbot

### 📌 AI Powered Assistant for Understanding Electricity Bills

Smart Meter Chatbot is an intelligent application that helps users analyze and understand their electricity consumption using Artificial Intelligence. The system reads energy bill PDFs, extracts important details, and allows users to interact in natural language to get insights such as total usage, charges, tariff details, and saving suggestions.

---

## 🚀 Features

* ✅ Upload and analyze smart meter / electricity bill PDF
* ✅ Automatic extraction of:

  * Units consumed
  * Billing amount
  * Tariff plan
  * Due date
  * Meter number & customer details
* ✅ Interactive AI chatbot
* ✅ Natural language queries supported
* ✅ Energy saving recommendations
* ✅ Simple, clean and user-friendly UI
* ✅ Secure API key management using `.env`

---

## 🖼 Screenshots
* Home Page
* PDF Upload
* Chatbot Interaction
* Bill Summary

## 🎥 Demo Workflow

1. User uploads electricity bill (PDF)
2. System extracts text and key fields
3. AI model processes data
4. User asks questions
5. Chatbot responds with bill insights

---

## 🧠 Sample Questions

You can ask:

* 💬 “What is my total bill amount?”
* 💬 “How many units did I consume?”
* 💬 “What is the due date?”
* 💬 “Why is my bill high this month?”
* 💬 “Give tips to reduce electricity usage”

---

## 🛠 Technology Stack

* **Python** – Core programming
* **Streamlit / Flask** – Frontend & API
* **LLM / AI Model** – Intelligent responses
* **PDF Parser** – Bill data extraction
* **dotenv** – Secure configuration

---

## 🧩 Architecture Diagram

```
User → Upload PDF  
        ↓  
PDF Processor → Extract Bill Data  
        ↓  
AI Model → Understand Query  
        ↓  
Chatbot → Response to User
```

---

## 📁 Project Structure

```
Smart Meter chatbot/
│
├── app.py                         # Main application  
├── requirements.txt               # Libraries  
├── .env                           # API configuration  
├── tmp_sample_energy_bill.pdf     # Sample file  
└── README.md                      # Documentation  
```

---

## ⚙ Installation & Setup

### 1. Clone Repository

```bash
git clone <your-github-repo-link>
cd Smart-Meter-chatbot
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create `.env` file:

```
API_KEY = your_api_key
MODEL = your_model_name
```

### 5. Run Project

```bash
python app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🔌 API Flow Explanation

1. **PDF Module**

   * Reads bill
   * Extracts text
   * Identifies key fields

2. **AI Module**

   * Understands user query
   * Matches with bill data
   * Generates response

3. **UI Layer**

   * Chat interface
   * Displays summary
   * Shows recommendations

---

## 🎯 Use Cases

* ✔ Household bill analysis
* ✔ Customer support automation
* ✔ Smart city solutions
* ✔ Energy consumption tracking
* ✔ Bill verification system

---

## 🚀 Future Enhancements

* 🔹 Monthly comparison graphs
* 🔹 Multi-language chatbot
* 🔹 Predict next bill
* 🔹 Real-time smart meter API
* 🔹 Mobile app version

---

## 🤝 Contribution

1. Fork repository
2. Create feature branch
3. Commit changes
4. Create pull request

---

## 📄 License

This project is under **MIT License**.

---

## 👩‍💻 Team Members

This project was developed as a group project by:

- **Sanika Muluk**
- **Om Chaudhari**
- **Tanvi Deshpande**
- **Aakanksha Naiknaware**
- **Anushka Bhale**
- **Rutuja Patwari**
- **Sarthak Kadam**

Under the guidance of  
**Prof.Kavita Kumavat**  
Department of Computer Engineering

### ⭐ If you found this helpful, please star this repository!
