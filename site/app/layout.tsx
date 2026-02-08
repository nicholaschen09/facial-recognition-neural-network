import "./globals.css";
import type { Metadata } from "next";
import { Lora } from "next/font/google";
import React from "react";

const lora = Lora({
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Facial Recognition Neural Network – Nicholas Chen",
  description:
    "Blog-style walkthrough of a PyTorch facial recognition system with preprocessing, CNN training, evaluation, and a real-time webcam demo.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={lora.className}>
      <body>{children}</body>
    </html>
  );
}


