**UPDATE**

# 🌱 Seed Travel MCH – Modular Curve-based Seed Interpolator

> ✨ **Nowa, autorska wersja skryptu „Seed Travel”** napisana od podstaw – lżejsza, modularna i gotowa do rozbudowy.

---

## 🚀 Czym jest Seed Travel MCH?

To alternatywny skrypt do interpolacji między seedami w Stable Diffusion WebUI (w zakładce `txt2img`). Generuje animacje z płynnymi przejściami pomiędzy obrazami na podstawie:

- `seed` i `subseed`
- rodzaju krzywej przejścia
- siły interpolacji (`subseed_strength`)

Wersja MCH została **napisana od zera**, z naciskiem na:

- **czytelność kodu**
- **modularność**
- **łatwość rozbudowy**
- **niezależna**

---

## 🧪 Tryby pracy

### 🔹 Seed Morph
Interpoluje między kolejnymi seedami `A -> B -> C`, używając `subseed_strength` w zakresie `0 → 1`.

> Efekt: Przejścia między kompletnie różnymi obrazami (np. różne postacie, sceny).

### 🔹 Subseed Morph (Nowość)
Zachowuje jeden seed i stopniowo zwiększa `subseed_strength` – tworząc animację zmieniającej się wersji jednego obrazu.

> Efekt: Jeden obraz „ewoluuje” w czasie. Idealny do efektów morph/tween.

---

## 🖼️ Przykładowe zastosowania

- Tworzenie płynnych przejść między wygenerowanymi scenami
- Renderowanie sekwencji jako klipów wideo `.mp4`
- Generowanie materiału bazowego do interpolacji RIFE (jeśli dodasz później)

---

## ⚙️ Instalacja

1. Wklej plik `seed_travel_mch.py` do folderu `extensions` lub `scripts` w Stable Diffusion WebUI.
2. Uruchom lub zrestartuj WebUI.
3. Przejdź do zakładki `txt2img` → rozwiń `Script` → wybierz `Seed Travel MCH - Blend Generator`.

---

## 📌 Wymagania

- Stable Diffusion WebUI (AUTOMATIC1111 lub fork kompatybilny)
- Python 3.10+
- `imageio`, `Pillow`, `numpy`, `torchvision`

---
