import SwiftUI

struct TapPulse: View {
    let center: CGPoint
    let pulseID: UUID

    @State private var scale: CGFloat = 0.35
    @State private var opacity = 1.0

    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.slopePrimary, lineWidth: 2)
                .frame(width: 28, height: 28)
                .scaleEffect(scale)
                .opacity(opacity)

            Circle()
                .fill(Color.slopePrimary)
                .frame(width: 8, height: 8)
                .shadow(color: Color.slopePrimary.opacity(0.35), radius: 0, x: 0, y: 0)
        }
        .position(center)
        .task(id: pulseID) {
            scale = 0.35
            opacity = 1
            withAnimation(.easeOut(duration: 0.8)) {
                scale = 3
                opacity = 0
            }
        }
        .allowsHitTesting(false)
    }
}

