import "./globals.css";
import type { Metadata } from "next";
import React from "react";

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
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}


