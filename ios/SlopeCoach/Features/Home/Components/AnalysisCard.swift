import SwiftUI

struct AnalysisCard: View {
    let analysis: AnalysisSummary

    var body: some View {
        HStack(spacing: 12) {
            Image(analysis.thumbnailName)
                .resizable()
                .scaledToFill()
                .frame(width: 100, height: 76)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(alignment: .bottomTrailing) {
                    Image(systemName: "play.fill")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.white)
                        .padding(6)
                        .background(.ultraThinMaterial, in: Circle())
                        .padding(5)
                }

            VStack(alignment: .leading, spacing: 5) {
                Text(analysis.title)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Color.slopeNavy)
                    .lineLimit(1)

                Text(analysis.timestamp)
                    .font(.system(size: 11, weight: .regular))
                    .foregroundStyle(.secondary)

                HStack(spacing: 8) {
                    StatusBadge(status: analysis.status)

                    statusDetail
                }
            }

            Spacer(minLength: 0)
        }
        .padding(10)
        .background(.white, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color.slopeBorder, lineWidth: 1)
        }
        .shadow(color: Color.slopeNavy.opacity(0.04), radius: 8, y: 3)
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder
    private var statusDetail: some View {
        switch analysis.status {
        case .ready(let score):
            Text("\(score) Score")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Color.slopeNavy)
        case .partial(let reason):
            Text(reason)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(Color.slopeWarning)
                .lineLimit(1)
        }
    }
}
