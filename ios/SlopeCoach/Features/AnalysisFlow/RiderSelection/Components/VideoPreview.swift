import SwiftUI

struct RiderVideoPreview: View {
    let imageName: String
    let sourceSize: CGSize
    let selectionState: RiderSelectionState
    let currentMask: RiderMask?
    let tapPoint: NormalizedVideoPoint?
    let pulseID: UUID
    let isScrubbing: Bool
    let onTap: (NormalizedVideoPoint) -> Void

    var body: some View {
        GeometryReader { proxy in
            let mapper = VideoCoordinateMapper(
                sourceSize: sourceSize,
                containerSize: proxy.size,
                contentMode: .aspectFit
            )

            ZStack {
                Color(red: 11 / 255, green: 21 / 255, blue: 32 / 255)

                Image(imageName)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                LinearGradient(
                    colors: [.black.opacity(0.2), .clear, .black.opacity(0.2)],
                    startPoint: .top,
                    endPoint: .bottom
                )

                if let currentMask, !isScrubbing, selectionState == .selected {
                    RiderMaskOverlay(mask: currentMask, sourceSize: sourceSize)
                }

                if let tapPoint, selectionState == .selecting {
                    TapPulse(center: mapper.displayPoint(for: tapPoint), pulseID: pulseID)
                }

                stateOverlay
            }
            .contentShape(Rectangle())
            .gesture(
                SpatialTapGesture()
                    .onEnded { value in
                        guard selectionState == .idle || selectionState == .failed,
                              let point = mapper.normalizedPoint(forDisplayPoint: value.location) else {
                            return
                        }
                        onTap(point)
                    }
            )
        }
        .aspectRatio(16 / 10, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color.white.opacity(0.12), lineWidth: 1)
        }
        .accessibilityLabel("Video preview")
        .accessibilityHint("Tap the skier you want to analyze")
    }

    @ViewBuilder
    private var stateOverlay: some View {
        switch selectionState {
        case .idle:
            hint(text: "Tap a skier", systemImage: "scope")
        case .selecting:
            VStack {
                Spacer()
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                        .tint(.white)
                    Text("Selecting rider...")
                        .font(.caption.weight(.medium))
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(.black.opacity(0.48), in: Capsule())
                .padding(.bottom, 12)
            }
        case .failed:
            hint(text: "Tap a clearer part of the skier", systemImage: "scope")
        case .selected, .lost:
            EmptyView()
        }
    }

    private func hint(text: String, systemImage: String) -> some View {
        Label(text, systemImage: systemImage)
            .font(.caption.weight(.medium))
            .foregroundStyle(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.black.opacity(0.38), in: Capsule())
    }
}

