import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)

    # Color palette
    c_input = "#4A90E2"
    c_enc = "#5C6BC0"
    c_dual = "#26A69A"
    c_spectral = "#AB47BC"
    c_latent = "#FFA726"
    c_dec = "#42A5F5"
    c_pixel = "#26C6DA"
    c_output = "#66BB6A"

    def draw_box(x, y, w, h, title, subtitle="", color="#E0E0E0", text_color="white"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=color, edgecolor="#2c3e50", linewidth=1.5,
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", color=text_color, fontsize=9.5, fontweight="bold")
            ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", color=text_color, fontsize=7.5)
        else:
            ax.text(x + w/2, y + h/2, title, ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor="#37474F", edgecolor="#37474F", width=1.5, headwidth=6, headlength=7, shrink=0.05)
        )
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.18, label, ha="center", va="bottom", fontsize=7.5, color="#37474F", fontweight="semibold")

    # Titles
    ax.text(7, 6.6, "LamyNet (Lite-DPAS v5) Architecture Overview", ha="center", va="center", fontsize=15, fontweight="bold", color="#1A237E")
    ax.text(7, 6.2, "Massive MIMO CSI Feedback Autoencoder with Dual-Axis Depthwise Conv & Bandwise Spectral Gate", ha="center", va="center", fontsize=9.5, color="#546E7A", style="italic")

    # Top Row: Encoder (Left to Right)
    ax.text(0.5, 5.3, "User Equipment (UE) Encoder", fontsize=11, fontweight="bold", color="#1565C0")
    
    # 1. Input CSI
    draw_box(0.5, 4.0, 1.4, 0.9, "CSI Tensor", "2 x 32 x 32\n(Real/Imag)", c_input)
    draw_arrow(1.9, 4.45, 2.3, 4.45)

    # 2. ConvNormAct
    draw_box(2.3, 4.0, 1.4, 0.9, "Initial Conv", "3x3 Conv + BN\n8 x 32 x 32", c_enc)
    draw_arrow(3.7, 4.45, 4.1, 4.45)

    # 3. DualAxisLiteBlock 1
    draw_box(4.1, 4.0, 1.6, 0.9, "DualAxis Lite", "3x3 & 1x15 DW\n8 x 32 x 32", c_dual)
    draw_arrow(5.7, 4.45, 6.1, 4.45)

    # 4. Learned Downsample
    draw_box(6.1, 4.0, 1.6, 0.9, "Learned Down", "Stride-2 DW\n8 x 16 x 16", c_enc)
    draw_arrow(7.7, 4.45, 8.1, 4.45)

    # 5. Bandwise Spectral Gate
    draw_box(8.1, 4.0, 1.6, 0.9, "Spectral Gate", "rFFT 4-Band\n8 x 16 x 16", c_spectral)
    draw_arrow(9.7, 4.45, 10.1, 4.45)

    # 6. Channel Slim + Flatten
    draw_box(10.1, 4.0, 1.5, 0.9, "Slim & Flat", "1x1 Conv (8->4)\n4 x 16 x 16 (1024)", c_enc)
    draw_arrow(11.6, 4.45, 12.0, 4.45)

    # 7. Linear Codeword
    draw_box(12.0, 3.8, 1.5, 1.3, "Codeword", "Linear FC\nDim: 2048/CR\n(512/128/64/32)", c_latent)

    # Feedback Link Arrow
    ax.annotate(
        "", xy=(12.75, 2.3), xytext=(12.75, 3.8),
        arrowprops=dict(facecolor="#E65100", edgecolor="#E65100", width=2.5, headwidth=8, headlength=9, linestyle="--")
    )
    ax.text(12.85, 3.0, "Uplink CSI Feedback\n(Limited Bits)", ha="left", va="center", fontsize=8, color="#BF360C", fontweight="bold")

    # Bottom Row: Decoder (Right to Left)
    ax.text(0.5, 2.6, "Base Station (BS) Decoder", fontsize=11, fontweight="bold", color="#00838F")

    # 8. FC Expansion
    draw_box(11.5, 1.2, 1.5, 0.9, "FC Expansion", "Linear FC\n4 x 16 x 16", c_dec)
    draw_arrow(11.5, 1.65, 11.0, 1.65)

    # 9. DualAxisLiteBlock 2
    draw_box(9.4, 1.2, 1.6, 0.9, "DualAxis Lite", "3x3 & 1x15 DW\n4 x 16 x 16", c_dual)
    draw_arrow(9.4, 1.65, 8.9, 1.65)

    # 10. Spectral Gate Decoder
    draw_box(7.3, 1.2, 1.6, 0.9, "Spectral Gate", "rFFT 4-Band\n8 x 16 x 16", c_spectral)
    draw_arrow(7.3, 1.65, 6.8, 1.65)

    # 11. PixelShuffle Upsample
    draw_box(4.9, 1.2, 1.9, 0.9, "PixelShuffle Up", "2x Sub-Pixel DW\n8 x 32 x 32", c_pixel)
    draw_arrow(4.9, 1.65, 4.4, 1.65)

    # 12. Final Reconstruction
    draw_box(2.8, 1.2, 1.6, 0.9, "Recon Conv", "3x3 Conv + Sigmoid\n2 x 32 x 32", c_dec)
    draw_arrow(2.8, 1.65, 2.3, 1.65)

    # 13. Output CSI
    draw_box(0.7, 1.2, 1.6, 0.9, "Reconstructed CSI", "2 x 32 x 32\n(Recovered H)", c_output)

    # Legend / Key concepts at bottom
    legend_y = 0.3
    ax.text(1.0, legend_y, "Key Innovations:", fontsize=8.5, fontweight="bold", color="#37474F")
    ax.text(2.6, legend_y, "• Dual-Axis DW: Spatial(3x3) + Long-Delay(1x15)", fontsize=8, color="#00695C")
    ax.text(6.8, legend_y, "• Bandwise Spectral Gate: Frequency rFFT Energy", fontsize=8, color="#6A1B9A")
    ax.text(10.8, legend_y, "• Spatial Preservation: 16x16 PixelShuffle", fontsize=8, color="#00838F")

    plt.tight_layout()
    plt.savefig("source/architecture_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Architecture diagram created successfully!")

if __name__ == "__main__":
    draw_architecture()
