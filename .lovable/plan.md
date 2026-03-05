

# SolarMind AI — Implementation Plan

## Overview
A futuristic AI-powered solar inverter monitoring dashboard with dark theme, neon accents, and glassmorphism cards. The app will include API/WebSocket service layers ready for FastAPI backend integration, with built-in mock data for demo mode.

## Design System
- **Dark slate background** with glassmorphism cards (backdrop-blur, semi-transparent borders)
- **Neon green** primary accent, **electric blue** secondary, **orange** warning, **red** critical
- Glowing effects on critical alerts, smooth animations throughout
- Futuristic monospace/sans-serif typography

## Pages

### 1. Landing Page
- Hero with animated title "SolarMind AI" and tagline "Predict. Explain. Act. Automatically."
- Feature cards (ML predictions, SHAP explainability, autonomous maintenance)
- System architecture diagram section
- AI capabilities showcase
- Live dashboard preview mockup
- CTA to enter dashboard

### 2. Dashboard Page
- **KPI Cards**: Total Inverters, Healthy, High Risk, Critical — with animated counters
- **Risk Heatmap**: Grid of inverter cells color-coded by risk score (green/yellow/orange/red), clickable to inverter details
- **Telemetry Charts**: Recharts line charts for temperature, efficiency, string mismatch (48hr window)
- **AI Diagnosis Panel**: Shows risk score, summary, root cause, recommended action, and causal drivers bar chart
- Real-time updates via WebSocket service layer

### 3. Inverter Details Page
- Risk score gauge visualization
- Telemetry trend charts
- SHAP waterfall chart (feature contributions)
- Delta-SHAP feature bar chart (change analysis)
- AI diagnostic report card

### 4. AI Assistant Page
- Chat interface with operator-style chat bubbles
- Input field to query the RAG system
- Displays AI responses with markdown rendering
- Example prompts shown as quick-action buttons

## Services Layer
- **API service** (`services/api.ts`): Axios-based methods for all REST endpoints (`getInverters`, `getInverterReport`, `getTelemetry`, `predictInverter`, `queryAI`) with mock data fallback
- **WebSocket service** (`services/websocket.ts`): Connection to `/ws/stream/{plant_id}`, dispatches real-time updates to dashboard state

## Layout
- **Navbar**: SolarMind logo, nav links, connection status indicator
- **Sidebar**: Collapsible navigation with icons for Dashboard, Inverters, Assistant
- Responsive layout for all screen sizes

## Mock Data
- 20 inverters with varied risk scores, telemetry histories, and diagnostic reports
- INV-14 highlighted as high-risk demo case (risk 0.82, rising temp, falling efficiency)
- Mock SHAP values and causal driver data

