import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CIRCUITRON",
  description:
    "Convert hand-drawn circuit diagrams into editable digital schematics.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
