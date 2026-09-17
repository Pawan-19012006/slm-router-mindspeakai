# TOY RESEARCH

## Toy – Proposed System Architecture

### 1. Core Concept

The toy will use a **Small Language Model (SLM)** for lightweight, frequently used interactions instead of running a large LLM locally.

The system will support both:

- **Local AI processing** for simple/common questions.
- **Cloud AI processing** through the parent's preferred AI provider for complex or new questions.

---

### 2. Basic Interaction Flow

**Child speaks → Speech Recognition → Query Understanding → Routing → Response → Text-to-Speech → Speaker**

The system should use **asynchronous/parallel processing and streaming wherever possible** to minimize response latency.

---

### 3. Local SLM

The SLM running on the toy will primarily act as an **intelligent router and lightweight response engine**.

It can identify:

- Simple questions
- Memory-related questions
- Questions that can be answered locally
- Questions requiring cloud AI

#### Example 1: Memory Query

> "What is my name?"

→ Identify as a **memory query** → retrieve stored name → generate short response.

#### Example 2: Complex Knowledge Query

> "Why does the moon follow me?"

→ Identify as a **complex knowledge query** → route to cloud AI.

---

### 4. Toy Memory

Memory should be maintained separately from the SLM rather than relying on the model itself to remember information.

Possible stored information:

- Child's name
- Preferences
- Previously provided information
- Important interaction context

#### Example

```text
"My favorite animal is a tiger."
        ↓
Memory Storage
        ↓
favorite_animal = tiger

Later:

"What is my favorite animal?"
        ↓
Memory Retrieval
        ↓
"Tiger"

The exact memory architecture is still to be investigated.

---

### 5. Cloud AI Integration

For questions that are beyond the local SLM's capabilities, the toy can use the **parent's AI provider** through the mobile application.

Possible providers include:

- OpenAI
- Google Gemini
- Anthropic Claude
- Other compatible AI services

The proposed architecture does **not require the company's own AI inference server**.

```text
Toy → Mobile App → Parent's AI API → Mobile App → Toy

The mobile application can manage the provider connection and API credentials.

---

### 6. Mobile Application

The mobile application acts as the **bridge and system orchestrator** between the toy and cloud AI.

It can handle:

- Toy pairing
- Memory management
- Cloud AI connectivity
- AI provider/API configuration
- Parental controls
- Settings
- Potentially STT/TTS and other heavier processing

The app should support **offline functionality where possible** and should not depend entirely on an internet connection.

---

### 7. Offline Mode

The system should remain functional without internet.

Possible offline architecture:

```text
Child → ESP32 → Local SLM/Memory → TTS → Speaker

The toy can continue handling:

- Basic questions
- Memory retrieval
- Predefined interactions
- Local SLM-supported responses

Complex cloud-dependent questions can fall back to a response such as:

> "I don't know that yet. Let's try something else!"

---

### 8. Connectivity Modes

| Mode | Functionality |
|---|---|
| **ESP32 only** | Local SLM + memory + basic interactions |
| **Toy + Phone, no Internet** | Local functionality + phone-side services |
| **Toy + Phone + Internet** | Local SLM + parent's cloud AI |
| **Cloud unavailable** | Automatically fall back to local capabilities |

**Wi-Fi is therefore not mandatory for the toy's basic functionality.**

Bluetooth/local connectivity can be used between the toy and mobile application.

---

### 9. Latency & Asynchronous Processing

Low latency is a major requirement.

Instead of:

```text
STT → Router → Cloud API → LLM → TTS → Audio

being completely sequential, the system should use:

- **Streaming audio**
- **Incremental speech recognition**
- **Asynchronous routing**
- **Streaming cloud responses**
- **Incremental TTS**
- **Background memory updates**

The objective is to allow the toy to **start responding as soon as enough information is available**, rather than waiting for every process to completely finish.

---

### 10. Prototype Hardware

The initial prototype can be built using an **ESP32/ESP32-S3 development board** along with:

- Microphone
- Speaker/audio amplifier
- Local SLM
- Connectivity to mobile application

Once the architecture and performance are validated, the system can be migrated to:

**Custom PCB → optimized embedded hardware → potentially Raspberry Pi/more powerful edge hardware if required.**

The software architecture should remain modular so that the underlying hardware can be upgraded later.

---

### 11. Some Useful References to Look Into

- [ESP32 AI – Running a Million Parameter Model Inside ESP32](https://github.com/slvDev/esp32-ai)
- [Espressif ESP-SR – Audio Frontend, Noise Suppression, VAD, Wake Word Detection, and Offline Speech Command Recognition](https://github.com/espressif/esp-sr)
- [Espressif ESP-DL – Quantization Support and Inference for ESP32](https://github.com/espressif/esp-dl)
- [ESP32-LLM – 3.3M Parameter INT8 Quantized Model](https://github.com/doryiii/esp32-llm)
- [Sherpa-ONNX – Speech/AI Framework Supporting Android, iOS, Raspberry Pi, and More](https://github.com/k2-fsa/sherpa-onnx)