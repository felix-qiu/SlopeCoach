import SwiftUI

struct RiderMaskOverlay: View {
    let mask: RiderMask
    let sourceSize: CGSize

    var body: some View {
        GeometryReader { proxy in
            let mapper = VideoCoordinateMapper(
                sourceSize: sourceSize,
                containerSize: proxy.size,
                contentMode: .aspectFit
            )
            let points = mapper.displayPoints(for: mask)

            ZStack {
                Canvas { context, _ in
                    guard let firstPoint = points.first else { return }
                    var path = Path()
                    path.move(to: firstPoint)
                    for point in points.dropFirst() {
                        path.addLine(to: point)
                    }
                    path.closeSubpath()

                    context.fill(path, with: .color(Color.slopePrimary.opacity(0.15)))
                    context.stroke(
                        path,
                        with: .color(Color.slopePrimary.opacity(0.9)),
                        style: StrokeStyle(lineWidth: 1.6, lineJoin: .round)
                    )
                }

                if let labelPosition = labelPosition(for: points, in: proxy.size) {
                    Text("✓ Selected")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.slopePrimary, in: Capsule())
                        .position(labelPosition)
                }
            }
        }
        .transition(.opacity.combined(with: .scale(scale: 0.96)))
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    private func labelPosition(for points: [CGPoint], in size: CGSize) -> CGPoint? {
        guard !points.isEmpty else { return nil }
        let minX = points.map(\.x).min() ?? 0
        let maxX = points.map(\.x).max() ?? 0
        let minY = points.map(\.y).min() ?? 0
        return CGPoint(
            x: min(max((minX + maxX) / 2, 46), size.width - 46),
            y: max(minY - 13, 14)
        )
    }
}

