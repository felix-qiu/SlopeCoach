import SwiftUI

struct VideoTimeline: View {
    let imageName: String
    let currentTime: TimeInterval
    let duration: TimeInterval
    let onScrubBegan: () -> Void
    let onScrubChanged: (TimeInterval) -> Void
    let onScrubEnded: () -> Void

    private let thumbnailWidth: CGFloat = 52
    private let thumbnailHeight: CGFloat = 60
    private let thumbnailCount = 36

    @State private var dragStartTime: TimeInterval?

    var body: some View {
        GeometryReader { proxy in
            let stripWidth = thumbnailWidth * CGFloat(thumbnailCount)
            let playheadX = proxy.size.width / 2
            let timeOffset = CGFloat(currentTime / max(duration, 0.001)) * stripWidth

            ZStack(alignment: .leading) {
                timelineStrip(stripWidth: stripWidth)
                    .offset(x: playheadX - timeOffset)

                playhead
                    .position(x: playheadX, y: proxy.size.height / 2)
            }
            .clipped()
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 1)
                    .onChanged { value in
                        let startTime: TimeInterval
                        if let dragStartTime {
                            startTime = dragStartTime
                        } else {
                            startTime = currentTime
                            dragStartTime = currentTime
                            onScrubBegan()
                        }

                        let timeDelta = TimeInterval(-value.translation.width / stripWidth) * duration
                        onScrubChanged(min(max(startTime + timeDelta, 0), duration))
                    }
                    .onEnded { _ in
                        dragStartTime = nil
                        onScrubEnded()
                    }
            )
        }
        .frame(height: 84)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Video timeline")
        .accessibilityValue(RiderSelectionTimeFormatter.string(from: currentTime))
    }

    private func timelineStrip(stripWidth: CGFloat) -> some View {
        VStack(spacing: 2) {
            ZStack(alignment: .leading) {
                ForEach(timeMarkers, id: \.self) { marker in
                    VStack(spacing: 2) {
                        Text(RiderSelectionTimeFormatter.string(from: marker))
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(Color(.secondaryLabel))
                            .monospacedDigit()
                        Rectangle()
                            .fill(Color(.separator))
                            .frame(width: 1, height: 4)
                    }
                    .position(
                        x: CGFloat(marker / max(duration, 0.001)) * stripWidth,
                        y: 10
                    )
                }
            }
            .frame(width: stripWidth, height: 20)

            HStack(spacing: 1) {
                ForEach(0..<thumbnailCount, id: \.self) { index in
                    Image(imageName)
                        .resizable()
                        .scaledToFill()
                        .frame(width: thumbnailWidth, height: thumbnailHeight)
                        .clipped()
                        .accessibilityHidden(true)
                        .overlay {
                            if index.isMultiple(of: 2) {
                                Color.black.opacity(0.035)
                            }
                        }
                }
            }
            .frame(width: stripWidth, alignment: .leading)
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        }
    }

    private var playhead: some View {
        VStack(spacing: 0) {
            Triangle()
                .fill(Color.slopePrimary)
                .frame(width: 10, height: 7)

            Rectangle()
                .fill(Color.slopePrimary)
                .frame(width: 2, height: 77)
        }
        .frame(width: 10, height: 84, alignment: .top)
        .allowsHitTesting(false)
    }

    private var timeMarkers: [TimeInterval] {
        stride(from: 0.0, through: duration, by: 6).map { $0 }
    }
}

private struct Triangle: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.midX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}

