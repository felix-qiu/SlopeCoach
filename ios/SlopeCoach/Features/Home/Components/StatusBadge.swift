import SwiftUI

struct StatusBadge: View {
    let status: AnalysisStatus

    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(tint)
                .frame(width: 6, height: 6)

            Text(title)
                .font(.system(size: 10, weight: .bold))
                .tracking(0.35)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 9)
        .padding(.vertical, 6)
        .background(tint.opacity(0.11), in: Capsule())
    }

    private var title: String {
        switch status {
        case .ready:
            "READY"
        case .partial:
            "PARTIAL"
        }
    }

    private var tint: Color {
        switch status {
        case .ready:
            .slopeSuccess
        case .partial:
            .slopeWarning
        }
    }
}
